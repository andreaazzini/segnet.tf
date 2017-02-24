import tensorflow as tf

def accuracy(logits, labels, batch_size):
  equal_pixels = tf.reduce_sum(tf.to_float(tf.equal(logits, labels)))
  total_pixels = batch_size * 224 * 224 * 3
  return equal_pixels / total_pixels

def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return tf.reduce_mean(cross_entropy, name='loss')
