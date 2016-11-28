from inputs import inputs
from tqdm import tqdm
from upsample import upsample

import convnet as cnn
import tensorflow as tf

tf.app.flags.DEFINE_string('train', './input/train.tfrecords', 'Train data')
tf.app.flags.DEFINE_string('train_labels', './input/train_labels.tfrecords', 'Train labels data')
tf.app.flags.DEFINE_string('train_logs', './logs/train', 'Log directory')
tf.app.flags.DEFINE_string('labels_file', '../labels', 'Labels file')

tf.app.flags.DEFINE_integer('batch', 64, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 500, 'Number of training iterations')

FLAGS = tf.app.flags.FLAGS

def conv(x, channels_shape, name):
  return cnn.conv(x, [3, 3], channels_shape, 1, name)

def deconv(x, channels_shape, name):
  return cnn.deconv(x, [3, 3], channels_shape, 1, name)

def pool(x):
  return cnn.max_pool(x, 2, 2)

def inference(images):
  with tf.variable_scope('pool1'):
    conv1 = conv(images, [3, 64], 'conv1')
    conv2 = conv(conv1, [64, 64], 'conv2')
    pool1, index1 = pool(conv2)

  with tf.variable_scope('pool2'):
    conv3 = conv(pool1, [64, 128], 'conv3')
    conv4 = conv(conv3, [128, 128], 'conv4')
    pool2, index2 = pool(conv4)

  with tf.variable_scope('pool3'):
    conv5 = conv(pool2, [128, 256], 'conv5')
    conv6 = conv(conv5, [256, 256], 'conv6')
    conv7 = conv(conv6, [256, 256], 'conv7')
    pool3, index3 = pool(conv7)

  with tf.variable_scope('pool4'):
    conv8 = conv(pool3, [256, 512], 'conv8')
    conv9 = conv(conv8, [512, 512], 'conv9')
    conv10 = conv(conv9, [512, 512], 'conv10')
    pool4, index4 = pool(conv10)

  with tf.variable_scope('pool5'):
    conv11 = conv(pool4, [512, 512], 'conv11')
    conv12 = conv(conv11, [512, 512], 'conv12')
    conv13 = conv(conv12, [512, 512], 'conv13')
    pool5, index5 = pool(conv13)

  with tf.variable_scope('unpool1'):
    unpool1 = upsample(pool5, index5)
    deconv1 = deconv(unpool1, [512, 512], 'deconv13')
    deconv2 = deconv(deconv1, [512, 512], 'deconv12')
    deconv3 = deconv(deconv2, [512, 512], 'deconv11')

  with tf.variable_scope('unpool2'):
    unpool2 = upsample(deconv3, index4)
    deconv4 = deconv(unpool2, [512, 512], 'deconv10')
    deconv5 = deconv(deconv4, [512, 512], 'deconv9')
    deconv6 = deconv(deconv5, [256, 512], 'deconv8')

  with tf.variable_scope('unpool3'):
    unpool3 = upsample(deconv6, index3)
    deconv7 = deconv(unpool3, [256, 256], 'deconv7')
    deconv8 = deconv(deconv7, [256, 256], 'deconv6')
    deconv9 = deconv(deconv8, [128, 256], 'deconv5')

  with tf.variable_scope('unpool4'):
    unpool4 = upsample(deconv9, index2)
    deconv10 = deconv(unpool4, [128, 128], 'deconv4')
    deconv11 = deconv(deconv10, [64, 128], 'deconv3')

  with tf.variable_scope('unpool5'):
    unpool5 = upsample(deconv11, index1)
    deconv12 = deconv(unpool5, [64, 64], 'deconv2')
    deconv13 = deconv(deconv12, [3, 64], 'deconv1')

  return deconv13

def train():
  images, labels = inputs(FLAGS.train, FLAGS.train_labels, FLAGS.batch)
  logits = inference(images)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='loss')
  tf.scalar_summary(loss.op.name, loss)

  optimizer = tf.train.AdamOptimizer(1e-04)
  train_step = optimizer.minimize(cross_entropy)

  init = tf.initialize_all_variables()
  sess = tf.InteractiveSession()
  summary = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.train_logs, sess.graph)
  sess.run(init)
  tf.train.start_queue_runners()

  for step in tqdm(range(FLAGS.steps + 1)):
    _, loss_value = sess.run([train_step, loss])

    if step % 10 == 0:
      print('Step %d: loss = %.4f' % (step, loss_value))
      summary_str = sess.run(summary)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_logs):
    tf.gfile.DeleteRecursively(FLAGS.train_logs)
  tf.gfile.MakeDirs(FLAGS.train_logs)
  train()

if __name__ == '__main__':
  tf.app.run()
