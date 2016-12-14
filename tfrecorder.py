from datetime import datetime
import os
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train', 'input/raw/train', 'Training images directory')
tf.app.flags.DEFINE_string('test', 'input/raw/test', 'Test images directory')
tf.app.flags.DEFINE_string('train_labels', 'input/raw/train-labels', 'Training label images directory')
tf.app.flags.DEFINE_string('test_labels', 'input/raw/test-labels', 'Test label images directory')
tf.app.flags.DEFINE_string('labels_file', 'labels', 'Labels file')
tf.app.flags.DEFINE_string('output', 'input', 'Output data directory')

tf.app.flags.DEFINE_boolean('encode_test_images', True, 'Encode test images')

tf.app.flags.DEFINE_integer('train_shards', 1, 'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_integer('test_shards', 1, 'Number of shards in test TFRecord files')
tf.app.flags.DEFINE_integer('threads', 1, 'Number of threads to preprocess the images')

FLAGS = tf.app.flags.FLAGS
IGNORE_FILENAMES = ['.DS_Store']


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width):
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': _bytes_feature(image_buffer)
  }))

  return example


class ImageCoder(object):
  def __init__(self):
    self._sess = tf.Session()
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=3)

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png, feed_dict={self._png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename, coder):
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  image = coder.decode_png(image_data)

  assert len(image.shape) == 3
  assert image.shape[2] == 3
  height, width, _ = image.shape

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards):
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    if num_shards == 1:
      output_filename = '%s.tfrecords' % name
    else:
      output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]

      if filename.split('/')[-1] in IGNORE_FILENAMES:
        continue

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _process_dataset(name, directory, num_shards):
  file_path = '%s/*' % directory
  filenames = tf.gfile.Glob(file_path)
  _process_image_files(name, filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.train_shards')
  assert not FLAGS.test_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.test_shards')
  print('Saving results to %s' % FLAGS.output)

  _process_dataset('train', FLAGS.train, FLAGS.train_shards)
  _process_dataset('train_labels', FLAGS.train_labels, FLAGS.train_shards)
  if FLAGS.encode_test_images:
    _process_dataset('test', FLAGS.test, FLAGS.test_shards)
    _process_dataset('test_labels', FLAGS.test_labels, FLAGS.test_shards)

if __name__ == '__main__':
  tf.app.run()
