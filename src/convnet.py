import tensorflow as tf

def conv(x, receptive_field_shape, channels_shape, stride, name):
  kernel_shape = receptive_field_shape + channels_shape
  bias_shape = [channels_shape[-1]]

  weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
  biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))
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
  conv = tf.nn.conv2d_transpose(x, weights, [batch_size, height, width, channels_shape[0]], [1, stride, stride, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(tf.contrib.layers.batch_norm(conv_bias))

def max_pool(x, size, stride, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding, name='maxpool')
