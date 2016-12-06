from inputs import inputs
from models import SegNetAutoencoder

import classifier
import config
import tensorflow as tf
import utils

test_file = utils.get_test_set(config.working_dataset)

tf.app.flags.DEFINE_string('test', test_file, 'Test data')
tf.app.flags.DEFINE_string('ckpt_dir', './ckpts', 'Train checkpoint directory')
# tf.app.flags.DEFINE_string('test_labels', './input/test_labels.tfrecords', 'Test labels data')
tf.app.flags.DEFINE_string('test_logs', './logs/test', 'Log directory')

tf.app.flags.DEFINE_integer('batch', 35, 'Batch size')

FLAGS = tf.app.flags.FLAGS

def accuracy(logits, labels):
  equal_pixels = tf.reduce_sum(tf.to_float(tf.equal(logits, labels)))
  total_pixels = tf.to_float(tf.reduce_prod(tf.shape(logits)))
  return equal_pixels / total_pixels

def test():
  #images, labels = inputs(FLAGS.batch, FLAGS.test, FLAGS.test_labels)
  images = inputs(FLAGS.batch, FLAGS.test)
  #one_hot_labels = classifier.one_hot(labels)

  autoencoder = SegNetAutoencoder(2, max_images=20)
  logits = autoencoder.inference(images)

  #accuracy_op = accuracy(logits, one_hot_labels)
  #tf.scalar_summary('accuracy', accuracy_op)

  saver = tf.train.Saver(tf.global_variables())
  summary = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.test_logs)

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

    if not (ckpt and ckpt.model_checkpoint_path):
      print('No checkpoint file found')
      return

    ckpt_path = ckpt.model_checkpoint_path
    saver.restore(sess, ckpt_path)

    #accuracy_value, summary_str = sess.run([accuracy_op, summary])
    summary_str = sess.run(summary)
    summary_writer.add_summary(summary_str)
    summary_writer.flush()

    coord.request_stop()
    coord.join(threads)

def main(argv=None):
  utils.restore_logs(FLAGS.test_logs)
  test()

if __name__ == '__main__':
  tf.app.run()
