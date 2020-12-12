import os
import time
import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs
import pdb
import data


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(n, x_dim, n_class, z_dim, n_samples):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_samples)
    h_from_z = tf.layers.dense(z, 500)
    y_logits = tf.zeros([n, n_class])
    y = bn.onehot_categorical("y", y_logits)
    h_from_y = tf.layers.dense(tf.cast(y, tf.float32), 500)
    h = tf.nn.relu(h_from_z + h_from_y)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="variational")
def qz_xy(x, y, z_dim, n_samples):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.cast(tf.concat([x, y], -1), tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_samples)
    return bn


@zs.reuse_variables("classifier")
def qy_x(x, n_class):
    h = tf.layers.dense(tf.cast(x, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    y_logits = tf.layers.dense(h, n_class)
    return y_logits


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = "data"
    train_dataset, validate_dataset, test_dataset = data.load_mnist_datasets(data_path)
    x_train = train_dataset._images
    y_train = train_dataset._labels
    x_valid = validate_dataset.images
    y_valid = validate_dataset._labels
    x_test = test_dataset._images
    y_test = test_dataset._labels
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    n_train, x_dim = x_train.shape
    n_class = 10

    # Define model parameters
    z_dim = 40

    # Define training/evaluation parameters
    lb_samples = 10
    beta = 1200.
    epochs = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    valid_iters = x_valid.shape[0] // test_batch_size
    test_iters = x_test.shape[0] // test_batch_size
    valid_freq = 10

    # Build the computation graph
    n = tf.placeholder(tf.int32, shape=[], name="n")
    n_samples = tf.placeholder(tf.int32, shape=[], name="n_samples")
    model = build_gen(n, x_dim, n_class, z_dim, n_samples)

    # Labeled
    x_ph = tf.placeholder(tf.float32, shape=[None, x_dim], name="x_l")
    x_ph = tf.cast(tf.less(tf.random_uniform(tf.shape(x_ph)), x_ph), tf.int32)
    y_ph = tf.placeholder(tf.int32, shape=[None, n_class], name="y_l")
    variational = qz_xy(x_ph, y_ph, z_dim, n_samples)

    lower_bound = tf.reduce_mean(
        zs.variational.elbo(model,
                            observed={"x": x_ph, "y": y_ph},
                            variational=variational,
                            axis=0))

    # Build classifier
    qy_logits = qy_x(x_ph, n_class)
    qy = tf.nn.softmax(qy_logits)
    pred_y = tf.argmax(qy, 1)
    acc = tf.reduce_sum(
        tf.cast(tf.equal(pred_y, tf.argmax(y_ph, 1)), tf.float32) / tf.cast(tf.shape(x_ph)[0], tf.float32))
    onehot_cat = zs.distributions.OnehotCategorical(qy_logits)
    log_qy_x = onehot_cat.log_prob(y_ph)
    classifier_cost = -beta * tf.reduce_mean(log_qy_x)

    # Gather gradients
    cost = -(lower_bound - classifier_cost) / 2.
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
    grads = optimizer.compute_gradients(cost)
    infer_op = optimizer.apply_gradients(grads)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            lbs, train_accs = [], []
            for t in range(iters):
                indices = np.random.randint(0, n_train, size=batch_size)
                x_batch = x_train[indices]
                y_batch = y_train[indices]
                _, lb, train_acc = sess.run(
                    [infer_op, lower_bound, acc],
                    feed_dict={x_ph: x_batch,
                               y_ph: y_batch,
                               n_samples: lb_samples,
                               n: batch_size})
                lbs.append(lb)
                train_accs.append(train_acc)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s), Lower bound = {},  Accuracy: {:.2f}%'.
                  format(epoch, time_epoch, np.mean(lbs), np.mean(train_accs) * 100.))

            if epoch % valid_freq == 0:
                time_valid = -time.time()
                valid_lls, valid_accs = [], []
                for t in range(valid_iters):
                    valid_x_batch = x_valid[t * test_batch_size: (t + 1) * test_batch_size]
                    valid_y_batch = y_valid[t * test_batch_size: (t + 1) * test_batch_size]
                    valid_ll, valid_acc = sess.run(
                        [lower_bound, acc],
                        feed_dict={x_ph: valid_x_batch,
                                   y_ph: valid_y_batch,
                                   n_samples: lb_samples,
                                   n: test_batch_size})
                    valid_lls.append(valid_ll)
                    valid_accs.append(valid_acc)
                time_valid += time.time()
                print('>>> VALID ({:.1f}s)'.format(time_valid))
                print('>> Valid lower bound = {}'.format(np.mean(valid_lls)))
                print('>> Valid accuracy: {:.2f}%'.format(100. * np.mean(valid_accs)))

        time_test = -time.time()
        test_lls, test_accs = [], []
        for t in range(test_iters):
            test_x_batch = x_test[t * test_batch_size: (t + 1) * test_batch_size]
            test_y_batch = y_test[t * test_batch_size: (t + 1) * test_batch_size]
            test_ll, test_acc = sess.run(
                [lower_bound, acc],
                feed_dict={x_ph: test_x_batch,
                           y_ph: test_y_batch,
                           n_samples: lb_samples,
                           n: test_batch_size})
            test_lls.append(test_ll)
            test_accs.append(test_acc)
        time_test += time.time()
        print('>>> TEST ({:.1f}s)'.format(time_test))
        print('>> Test lower bound = {}'.format(np.mean(test_lls)))
        print('>> Test accuracy: {:.2f}%'.format(100. * np.mean(test_accs)))


if __name__ == "__main__":
    main()
