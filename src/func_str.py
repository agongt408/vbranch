fill = """

def func_%d(x):
    import tensorflow as tf
    shape = %s
    x = tf.gather_nd(x, %s)
    print '1' , x.get_shape().as_list()
    x = tf.reshape(x, shape[:-1] + (-1, %d))
    print '2' , x.get_shape().as_list()
    x = tf.tile(x, [1]*len(shape) + [%d])[%s]
    print '3' , x.get_shape().as_list()
    x = tf.pad(x, [%s])
    print '4' , x.get_shape().as_list()
    return tf.reshape(x, shape)
"""

fill_concat = """

def func_concat_%d(x):
    import tensorflow as tf
    shape = %s
    pool = tf.gather_nd(x, %s)
    print '1' , pool.get_shape().as_list()
    pool = tf.reshape(pool, shape[:-1] + (-1, %d))
    print '2' , pool.get_shape().as_list()
    pool = tf.tile(pool, [1]*len(shape) + [%d])[%s]
    print '3' , pool.get_shape().as_list()
    pool = tf.pad(pool, [%s])
    print '4' , pool.get_shape().as_list()
    pool = tf.reshape(pool, shape[:-1] + (-1,))

    conv = tf.gather_nd(x, %s)
    print '5' , conv.get_shape().as_list()
    conv = tf.reshape(conv, shape[:-1] + (-1, %d))
    print '6' , conv.get_shape().as_list()
    conv = tf.tile(conv, [1]*len(shape) + [%d])[%s]
    print '7' , conv.get_shape().as_list()
    conv = tf.pad(conv, [%s])
    print '8' , conv.get_shape().as_list()
    conv = tf.reshape(conv, shape[:-1] + (-1,))

    return tf.reshape(tf.concat([pool, conv], -1), shape)
"""

reshape = """

def func_reshape(x):
    import tensorflow as tf
    return tf.reshape(x, %s)
"""


