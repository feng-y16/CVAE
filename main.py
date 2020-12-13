import os
import time
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import zhusuan as zs
import urllib
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='default cvae model')
    parser.add_argument('--dataset-path', default='/data', help='Path of dataset')
    parser.add_argument('--download', action='store_true', default=False, help='Whether to download the data anyway')
    parser.add_argument('--seed', type=int, default=1234, help='Seed of the experiment')
    parser.add_argument('--save-path', type=str, default='/results', help='Where to dump the results')
    parser.add_argument('--n-generate', type=int, default=10, help='Number of generated images for each class')
    parser.add_argument('--batch-size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=500, help='Batch size for testing')
    parser.add_argument('--n-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--z-dim', type=int, default=40, help='Dimension of z')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--test-interval', type=int, default=1, help='Interval of doing testing')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples for MC')
    args = parser.parse_args()
    args.dataset_path = os.path.dirname(os.path.realpath(__file__)) + args.dataset_path
    args.save_path = os.path.dirname(os.path.realpath(__file__)) + args.save_path
    if not os.path.isdir(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    return args


def download_dataset(args, url, path):
    if not os.path.exists(path) or args.download:
        print('Downloading data from %s' % url)
        urllib.request.urlretrieve(url, path)
    else:
        print('File already exists.')


def load_mnist_datasets(args, one_hot=True):
    """
    Loads the MNIST dataset.

    :param args: arguments.
    :param one_hot: Whether to use one-hot representation for the labels.

    :return: The MNIST dataset.
    """
    path = args.dataset_path
    download_dataset(args, 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                     path + '/train-images-idx3-ubyte.gz')
    download_dataset(args, 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                     path + '/train-labels-idx1-ubyte.gz')
    download_dataset(args, 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                     path + '/t10k-images-idx3-ubyte.gz')
    download_dataset(args, 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                     path + '/t10k-labels-idx1-ubyte.gz')
    datasets = input_data.read_data_sets(path, one_hot=one_hot)
    return datasets


@zs.meta_bayesian_net(scope='gen', reuse_variables=True)
def build_gen(n, x_dim, n_class, z_dim, n_samples):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal('z', z_mean, std=1., group_ndims=1, n_samples=n_samples)
    h_from_z = tf.layers.dense(z, 500)
    y_logits = tf.zeros([n, n_class])
    y = bn.onehot_categorical('y', y_logits)
    h_from_y = tf.layers.dense(tf.cast(y, tf.float32), 500)
    h = tf.nn.relu(h_from_z + h_from_y)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.deterministic('x_out', tf.sigmoid(x_logits))
    bn.bernoulli('x', x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope='variational')
def qz_xy(x, y, z_dim, n_samples):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.cast(tf.concat([x, y], -1), tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal('z', z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_samples)
    return bn


def plot_lbs(args, all_lbs_mean, all_lbs_std):
    train_mean = all_lbs_mean[0, :]
    train_std = all_lbs_std[0, :]
    train_epochs = np.arange(1, len(train_mean) + 1)
    test_mean = all_lbs_mean[1, args.test_interval - 1:: args.test_interval]
    test_std = all_lbs_std[1, args.test_interval - 1:: args.test_interval]
    test_epochs = args.test_interval * np.arange(1, len(test_mean) + 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_epochs, train_mean, label='Training')
    ax.plot(test_epochs, test_mean, label='Testing')
    ax.fill_between(train_epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.fill_between(test_epochs, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ELBO')
    ax.legend(loc='best')
    plt.savefig(args.save_path + '/ELBO.pdf')
    plt.close()


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
    x_ph = tf.placeholder(tf.float32, shape=[None, x_dim], name='x_in')
    x_transformed = tf.cast(tf.less(tf.random_uniform(tf.shape(x_ph)), x_ph), tf.int32)
    y_ph = tf.placeholder(tf.int32, shape=[None, args.n_classes], name='y_in')
    variational = qz_xy(x_transformed, y_ph, args.z_dim, n_samples)
    lower_bound = tf.reduce_mean(zs.variational.elbo(model, observed={'x': x_transformed, 'y': y_ph},
                                                     variational=variational, axis=0))

    # Gather gradients
    cost = -lower_bound
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
    grads = optimizer.compute_gradients(cost)
    infer_op = optimizer.apply_gradients(grads)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_gen = np.zeros((args.n_classes * args.n_generate, args.n_classes))
        for i in range(args.n_classes):
            y_gen[i * args.n_generate: (i + 1) * args.n_generate, i] = 1
        bar = tqdm(total=args.epochs)
        postfix = None
        all_lbs_mean = np.zeros((2, args.epochs))
        all_lbs_std = np.zeros((2, args.epochs))
        for epoch in range(1, args.epochs + 1):
            bar.set_description(f'Epoch {epoch}')
            time_epoch = - time.time()
            lbs = []
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
            all_lbs_mean[0, epoch - 1] = np.mean(np.array(lbs))
            all_lbs_std[0, epoch - 1] = np.std(np.array(lbs))
            data = {'time/s': '{:.2f}'.format(time_epoch),
                    'lower bound': '{:.2f}'.format(all_lbs_mean[0, epoch - 1])}
            if postfix is None:
                postfix = data
            else:
                for key in data.keys():
                    postfix[key] = data[key]
            bar.set_postfix(postfix)
            if epoch % args.test_interval == 0 or epoch == args.epochs:
                time_test = - time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * args.test_batch_size: (t + 1) * args.test_batch_size]
                    test_y_batch = y_test[t * args.test_batch_size: (t + 1) * args.test_batch_size]
                    test_lb = sess.run([lower_bound], feed_dict={x_transformed: test_x_batch,
                                                                 y_ph: test_y_batch,
                                                                 n_samples: args.n_samples,
                                                                 n: args.test_batch_size})
                    test_lbs.append(test_lb)
                time_test += time.time()
                all_lbs_mean[1, epoch - 1] = np.mean(np.array(test_lbs))
                all_lbs_std[1, epoch - 1] = np.std(np.array(test_lbs))
                data = {'test time/s': '{:.2f}'.format(time_test),
                        'test lower bound': '{:.2f}'.format(all_lbs_mean[1, epoch - 1])}
                for key in data.keys():
                    postfix[key] = data[key]
                bar.set_postfix(postfix)
                bn_gen = model.observe(y=y_gen)
                x_out = tf.reshape(bn_gen['x_out'], [-1, 28, 28])
                x_gen = sess.run(x_out, feed_dict={n_samples: 1, n: args.n_classes * args.n_generate})
                x_gen = x_gen.reshape((-1, args.n_classes, 28, 28))
                x_gen = x_gen.transpose((0, 2, 1, 3))
                x_gen = 255 - 255 * x_gen.reshape((args.n_generate * 28, args.n_classes * 28))
                image = Image.fromarray(x_gen.astype('uint8'))
                image.save(args.save_path + f'/epoch{epoch}.png')
                plot_lbs(args, all_lbs_mean[:, 0: epoch], all_lbs_std[:, 0: epoch])
            bar.update(1)
        bar.close()


if __name__ == '__main__':
    plt.rcParams.update({'figure.autolayout': True})
    main(parse_args())
