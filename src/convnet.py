import tensorflow as tf

def conv(x, receptive_field_shape, channels_shape, stride, name, repad=False):
  kernel_shape = receptive_field_shape + channels_shape
  bias_shape = [channels_shape[-1]]

  weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
  biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))

  if repad:
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
    conv = tf.nn.conv2d(padded, weights, strides=[1, stride, stride, 1], padding='VALID')
  else:
    conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')

  conv_bias = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(tf.contrib.layers.batch_norm(conv_bias))

def deconv(x, receptive_field_shape, channels_shape, stride, name):
  kernel_shape = receptive_field_shape + channels_shape
  bias_shape = [channels_shape[0]]

  input_shape = x.get_shape().as_list()
  batch_size = input_shape[0]
  height = input_shape[1]
  width = input_shape[2]

  weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
  biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))
  conv = tf.nn.conv2d_transpose(x, weights, [batch_size, height * stride, width * stride, channels_shape[0]], [1, stride, stride, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(tf.contrib.layers.batch_norm(conv_bias))

def max_pool(x, size, stride, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding, name='maxpool')

def unpool(x, size):
  out = tf.concat_v2([x, tf.zeros_like(x)], 3)
  out = tf.concat_v2([out, tf.zeros_like(out)], 2)

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
    out_size = [-1, sh[1] * size, sh[2] * size, sh[3]]
    return tf.reshape(out, out_size)

  shv = tf.shape(x)
  ret = tf.reshape(out, tf.stack([-1, shv[1] * size, shv[2] * size, sh[3]]))
  ret.set_shape([None, None, None, sh[3]])
  return ret
