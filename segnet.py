from inputs import inputs
from tqdm import tqdm
from upsample import upsample

import classifier
import convnet as cnn
import initializer
import tensorflow as tf
import utils

tf.app.flags.DEFINE_string('train', './input/train.tfrecords', 'Train data')
tf.app.flags.DEFINE_string('train_ckpt', './ckpts/model.ckpt', 'Train checkpoint file')
tf.app.flags.DEFINE_string('train_labels', './input/train_labels.tfrecords', 'Train labels data')
tf.app.flags.DEFINE_string('train_logs', './logs/train', 'Log directory')
tf.app.flags.DEFINE_string('labels_file', '../labels', 'Labels file')

tf.app.flags.DEFINE_integer('batch', 12, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 40000, 'Number of training iterations')

FLAGS = tf.app.flags.FLAGS

def conv(x, channels_shape, name):
  return cnn.conv(x, [3, 3], channels_shape, 1, name)

def deconv(x, channels_shape, name):
  return cnn.deconv(x, [3, 3], channels_shape, 1, name)

def pool(x):
  return cnn.max_pool(x, 2, 2)


class SegNetAutoencoder:
  def __init__(self):
    self.indices = [None for i in range(5)]
    self.params = []

  def encode(self, images):
    tf.image_summary('input', images)

    with tf.variable_scope('pool1'):
      conv1 = conv(images, [3, 64], 'conv1_1')
      conv2 = conv(conv1, [64, 64], 'conv1_2')
      pool1, self.indices[0] = pool(conv2)

    with tf.variable_scope('pool2'):
      conv3 = conv(pool1, [64, 128], 'conv2_1')
      conv4 = conv(conv3, [128, 128], 'conv2_2')
      pool2, self.indices[1] = pool(conv4)

    with tf.variable_scope('pool3'):
      conv5 = conv(pool2, [128, 256], 'conv3_1')
      conv6 = conv(conv5, [256, 256], 'conv3_2')
      conv7 = conv(conv6, [256, 256], 'conv3_3')
      pool3, self.indices[2] = pool(conv7)

    with tf.variable_scope('pool4'):
      conv8 = conv(pool3, [256, 512], 'conv4_1')
      conv9 = conv(conv8, [512, 512], 'conv4_2')
      conv10 = conv(conv9, [512, 512], 'conv4_3')
      pool4, self.indices[3] = pool(conv10)

    with tf.variable_scope('pool5'):
      conv11 = conv(pool4, [512, 512], 'conv5_1')
      conv12 = conv(conv11, [512, 512], 'conv5_2')
      conv13 = conv(conv12, [512, 512], 'conv5_3')
      pool5, self.indices[4] = pool(conv13)

    return pool5

  def decode(self, code):
    with tf.variable_scope('unpool1'):
      unpool1 = upsample(code, self.indices[4])
      deconv1 = deconv(unpool1, [512, 512], 'deconv5_3')
      deconv2 = deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      unpool2 = upsample(deconv3, self.indices[3])
      deconv4 = deconv(unpool2, [512, 512], 'deconv4_3')
      deconv5 = deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      unpool3 = upsample(deconv6, self.indices[2])
      deconv7 = deconv(unpool3, [256, 256], 'deconv3_3')
      deconv8 = deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      unpool4 = upsample(deconv9, self.indices[1])
      deconv10 = deconv(unpool4, [128, 128], 'deconv2_2')
      deconv11 = deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      unpool5 = upsample(deconv11, self.indices[0])
      deconv12 = deconv(unpool5, [64, 64], 'deconv1_2')
      deconv13 = deconv(deconv12, [32, 64], 'deconv1_1')

    rgb_output = classifier.rgb(deconv13)
    tf.image_summary('output', rgb_output)

    return deconv13

  def prepare_encoder_parameters(self):
    param_format = 'conv%d_%d_%s'
    conv_layers = [2, 2, 3, 3, 3]

    for pool in range(1, 6):
      with tf.variable_scope('pool%d' % pool, reuse=True):
        for conv in range(1, conv_layers[pool - 1] + 1):
          weights = tf.get_variable(param_format % (pool, conv, 'W'))
          biases = tf.get_variable(param_format % (pool, conv, 'b'))
          self.params += [weights, biases]

  def get_encoder_parameters(self):
    return self.params


def inference(autoencoder, images):
  code = autoencoder.encode(images)
  autoencoder.prepare_encoder_parameters()
  return autoencoder.decode(code)

def train():
  autoencoder = SegNetAutoencoder()
  images, labels = inputs(FLAGS.train, FLAGS.train_labels, FLAGS.batch)
  one_hot_labels = classifier.one_hot(labels)

  logits = inference(autoencoder, images)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='loss')
  tf.scalar_summary(loss.op.name, loss)

  optimizer = tf.train.AdamOptimizer(1e-04)
  train_step = optimizer.minimize(cross_entropy)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    sess.run(init)
    initializer.initialize(autoencoder.get_encoder_parameters(), sess)

    summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_logs, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in tqdm(range(FLAGS.steps + 1)):
      sess.run(train_step)

      if step % 10 == 0:
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      if step % 1000 == 0:
        saver.save(sess, FLAGS.train_ckpt)

    coord.request_stop()
    coord.join(threads)

def main(argv=None):
  utils.restore_logs(FLAGS.train_logs)

  with tf.device('/gpu:0'):
    train()

if __name__ == '__main__':
  tf.app.run()
