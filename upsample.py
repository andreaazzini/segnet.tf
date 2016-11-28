from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import tensorflow as tf

@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad_with_argmax(
    op.inputs[0],
    grad,
    op.outputs[1],
    op.get_attr("ksize"),
    op.get_attr("strides"),
    padding=op.get_attr("padding")
  )

def unravel(argmax, shape):
  d = shape[2] * shape[3]
  return tf.pack([argmax // d, argmax % d // shape[3]])

def upsample(bottom, argmax):
  bottom_shape = tf.shape(bottom)
  batch_size = bottom_shape[0]
  height = bottom_shape[1] * 2
  width = bottom_shape[2] * 2
  channels = bottom_shape[3]
  top_shape = [batch_size, height, width, channels]
  argmax_shape = tf.to_int64(top_shape)
  argmax = unravel(argmax, argmax_shape)

  t1 = tf.to_int64(tf.range(channels))
  t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
  t1 = tf.reshape(t1, [-1, channels])
  t1 = tf.transpose(t1, perm=[1, 0])
  t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
  t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

  t2 = tf.to_int64(tf.range(batch_size))
  t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
  t2 = tf.reshape(t2, [-1, batch_size])
  t2 = tf.transpose(t2, perm=[1, 0])
  t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

  t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

  t = tf.concat(4, [t2, t3, t1])
  indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

  x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
  values = tf.reshape(x1, [-1])

  delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
  return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
