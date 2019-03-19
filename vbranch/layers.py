# Build layers

import tensorflow as tf

def fc_layer(x, n_in, n_out, name, use_bias=True, epsilon=1e-5):
    w = tf.get_variable(name + '_w', shape=[n_in, n_out])

    if use_bias:
        b = tf.get_variable(name + '_b', shape=[n_out])
        z = tf.nn.xw_plus_b(x, w, b)
    else:
        z = tf.matmul(x, w)

    batch_mean, batch_var = tf.nn.moments(z, [0])
    # scale and beta (shift)
    scale = tf.get_variable(name + '_bn_scale', initializer=tf.ones([n_out]))
    beta = tf.get_variable(name + '_bn_beta', initializer=tf.zeros([n_out]))
    z_hat = tf.nn.batch_normalization(z,batch_mean,batch_var,beta,scale,epsilon)

    output = tf.nn.relu(z_hat)
    return output
