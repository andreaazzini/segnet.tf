from inputs import inputs
from models import SegNetAutoencoder
from tqdm import tqdm

import classifier
import config
import initializer
import tensorflow as tf
import utils

train_file, train_labels_file = utils.get_training_set(config.working_dataset)

tf.app.flags.DEFINE_string('train', train_file, 'Train data')
tf.app.flags.DEFINE_string('train_ckpt', './ckpts/model.ckpt', 'Train checkpoint file')
tf.app.flags.DEFINE_string('ckpt_dir', './ckpts', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('train_labels', train_labels_file, 'Train labels data')
tf.app.flags.DEFINE_string('train_logs', './logs/train', 'Log directory')

tf.app.flags.DEFINE_integer('batch', 12, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 200, 'Number of training iterations')

FLAGS = tf.app.flags.FLAGS

def accuracy(logits, labels):
  equal_pixels = tf.reduce_sum(tf.to_float(tf.equal(logits, labels)))
  total_pixels = FLAGS.batch * 224 * 224 * 3
  return equal_pixels / total_pixels

def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
  return tf.reduce_mean(cross_entropy, name='loss')

def train():
  images, labels = inputs(FLAGS.batch, FLAGS.train, FLAGS.train_labels)
  one_hot_labels = classifier.one_hot(labels)

  autoencoder = SegNetAutoencoder(2)
  logits = autoencoder.inference(images)

  accuracy_op = accuracy(logits, one_hot_labels)
  loss_op = loss(logits, one_hot_labels)
  tf.scalar_summary('accuracy', accuracy_op)
  tf.scalar_summary(loss_op.op.name, loss_op)

  optimizer = tf.train.AdamOptimizer(1e-04)
  train_step = optimizer.minimize(loss_op)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
  config.gpu_options.allow_growth = True

  ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

    if not ckpt:
      print('No checkpoint file found. Initializing...')
      global_step = 0
      sess.run(init)
      # initializer.initialize(autoencoder.get_encoder_parameters(), sess)
    else:
      global_step = len(ckpt.all_model_checkpoint_paths) * FLAGS.steps
      ckpt_path = ckpt.model_checkpoint_path
      saver.restore(sess, ckpt_path)

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

      if step % FLAGS.batch == 0:
        saver.save(sess, FLAGS.train_ckpt, global_step=global_step)

    coord.request_stop()
    coord.join(threads)

def main(argv=None):
  utils.restore_logs(FLAGS.train_logs)
  train()

if __name__ == '__main__':
  tf.app.run()
