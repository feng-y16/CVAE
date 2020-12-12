import os
import time
import numpy as np
import argparse
import tensorflow as tf
from data import *
from cvae import *
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='default cvae model')
    parser.add_argument('--dataset-path', default='/data', help='Path of dataset')
    parser.add_argument('--download', action='store_true', default=False, help='Whether to download the data anyway')
    parser.add_argument('--seed', type=int, default=1234, help='Seed of the experiment')
    parser.add_argument('--save_path', type=str, default='/results', help='Where to dump the results')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--n-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--z-dim', type=int, default=40, help='Dimension of z')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--test-interval', type=int, default=10, help='Interval of doing testing')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples for MC')
    args = parser.parse_args()
    args.dataset_path = os.path.dirname(os.path.realpath(__file__)) + args.dataset_path
    args.save_path = os.path.dirname(os.path.realpath(__file__)) + args.save_path
    if not os.path.isdir(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if args.seed is not None:
        np.random.seed(args.seed)
    return args


def main(args):
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Load MNIST
    train_dataset, validate_dataset, test_dataset = load_mnist_datasets(args, one_hot=True)
    x_train = np.concatenate((train_dataset._images, validate_dataset.images), axis=0)
    y_train = np.concatenate((train_dataset._labels, validate_dataset._labels), axis=0)
    x_test = test_dataset._images
    y_test = test_dataset._labels
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    n_train, x_dim = x_train.shape
    iters = x_train.shape[0] // args.batch_size
    test_iters = x_test.shape[0] // args.test_batch_size

    # Build the computation graph
    n = tf.placeholder(tf.int32, shape=[], name='n')
    n_samples = tf.placeholder(tf.int32, shape=[], name='n_samples')
    model = build_gen(n, x_dim, args.n_classes, args.z_dim, n_samples)

    # Labeled
    x_ph = tf.placeholder(tf.float32, shape=[None, x_dim], name='x_in')
    x_ph = tf.cast(tf.less(tf.random_uniform(tf.shape(x_ph)), x_ph), tf.int32)
    y_ph = tf.placeholder(tf.int32, shape=[None, args.n_classes], name='y_in')
    variational = qz_xy(x_ph, y_ph, args.z_dim, n_samples)
    lower_bound = tf.reduce_mean(zs.variational.elbo(model, observed={'x': x_ph, 'y': y_ph},
                                                     variational=variational, axis=0))

    # Gather gradients
    cost = -lower_bound
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
    grads = optimizer.compute_gradients(cost)
    infer_op = optimizer.apply_gradients(grads)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, args.epochs + 1):
            time_epoch = -time.time()
            lbs, train_accs = [], []
            for t in range(iters):
                indices = np.random.randint(0, n_train, size=args.batch_size)
                x_batch = x_train[indices]
                y_batch = y_train[indices]
                _, lb = sess.run([infer_op, lower_bound], feed_dict={x_ph: x_batch,
                                                                     y_ph: y_batch,
                                                                     n_samples: args.n_samples,
                                                                     n: args.batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s), Lower bound = {}'.format(epoch, time_epoch, np.mean(lbs)))

            if epoch % args.test_interval == 0 or epoch == args.epochs:
                time_test = -time.time()
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * args.test_batch_size: (t + 1) * args.test_batch_size]
                    test_y_batch = y_test[t * args.test_batch_size: (t + 1) * args.test_batch_size]
                    test_ll = sess.run([lower_bound], feed_dict={x_ph: test_x_batch,
                                                                 y_ph: test_y_batch,
                                                                 n_samples: args.n_samples,
                                                                 n: args.test_batch_size})
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lls)))


if __name__ == '__main__':
    main(parse_args())
