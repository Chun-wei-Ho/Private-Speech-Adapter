import numpy as np
import tensorflow as tf

def vbkt_loss(target, inputs):
    dim = inputs.get_shape().as_list()[1] // 2

    inputs_mu = inputs[:, :dim, :, :]
    inputs_logsigma = inputs[:, dim:, :, :]
    target_mu = target[:, :dim, :, :]
    target_logsigma = target[:, dim:, :, :]

    inputs_mu = tf.reshape(inputs_mu, [tf.shape(inputs_mu)[0], -1])
    inputs_logsigma = tf.reshape(inputs_logsigma, [tf.shape(inputs_logsigma)[0], -1])
    target_mu = tf.reshape(target_mu, [tf.shape(target_mu)[0], -1])
    target_logsigma = tf.reshape(target_logsigma, [tf.shape(target_logsigma)[0], -1])

    loss = tf.reduce_mean(0.5 * tf.math.square(inputs_mu - target_mu) / (tf.math.exp(2.0 * target_logsigma) + 1e-8))
    return loss
def poly_kernel(x, y):
    x = tf.expand_dims(x, 0)
    y = tf.expand_dims(y, 1)
    res = tf.pow(tf.reduce_sum(x * y, axis=-1), 2)
    return res

def nst_loss(target, inputs):
    target = tf.reshape(target, [tf.shape(target)[0], tf.shape(target)[1], -1])
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], -1])

    target = tf.nn.l2_normalize(target, 2)
    inputs = tf.nn.l2_normalize(inputs, 2)

    loss = tf.reduce_mean(poly_kernel(target, target)) \
            + tf.reduce_mean(poly_kernel(inputs, inputs)) \
            - 2 * tf.reduce_mean(poly_kernel(target, inputs)) 
    return loss

kt_loss = dict(
    vbkt=vbkt_loss,
    nst=nst_loss,
    mse="mean_squared_error"
)