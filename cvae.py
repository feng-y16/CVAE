import tensorflow as tf
import zhusuan as zs


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
