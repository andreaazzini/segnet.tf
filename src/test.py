from inputs import inputs
from models import SegNetAutoencoder
from scalar_ops import accuracy, loss

import classifier
import config
import tensorflow as tf
import utils

test_file, test_labels_file = utils.get_test_set(config.working_dataset, include_labels=True)

tf.app.flags.DEFINE_string('ckpt_dir', './ckpts', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('test', test_file, 'Test data')
tf.app.flags.DEFINE_string('test_labels', test_labels_file, 'Test labels data')
tf.app.flags.DEFINE_string('test_logs', './logs/test', 'Log directory')

tf.app.flags.DEFINE_boolean('strided', True, 'Use strided convolutions and deconvolutions')

tf.app.flags.DEFINE_integer('batch', 200, 'Batch size')

FLAGS = tf.app.flags.FLAGS

def test():
  images, labels = inputs(FLAGS.batch, FLAGS.test, FLAGS.test_labels)
  tf.summary.image('labels', labels)
  one_hot_labels = classifier.one_hot(labels)

  autoencoder = SegNetAutoencoder(4, strided=FLAGS.strided)
  logits = autoencoder.inference(images)

  accuracy_op = accuracy(logits, one_hot_labels, FLAGS.batch)
  tf.scalar_summary('accuracy', accuracy_op)

  saver = tf.train.Saver(tf.global_variables())
  summary = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(FLAGS.test_logs)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
  session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  with tf.Session(config=session_config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

    if not (ckpt and ckpt.model_checkpoint_path):
      print('No checkpoint file found')
      return

    ckpt_path = ckpt.model_checkpoint_path
    saver.restore(sess, ckpt_path)

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
