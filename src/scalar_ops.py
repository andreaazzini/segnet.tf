from classifier import color_mask

import tensorflow as tf

def accuracy(logits, labels):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, 3)
  
  shape = logits.get_shape().as_list()
  n = shape[3]
  
  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  equal_pixels = tf.reduce_sum(tf.to_float(color_mask(one_hot, labels)))
  total_pixels = reduce(lambda x, y: x * y, shape[:3])
  return equal_pixels / total_pixels

def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return tf.reduce_mean(cross_entropy, name='loss')
