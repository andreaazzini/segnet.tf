import config
import os
import tensorflow as tf

def colors_of_dataset(dataset_name):
  return config.colors[dataset_name]

def get_dataset(dataset_name, include_labels, kind):
  path = os.path.join('input', dataset_name)
  data_binary_path = os.path.join(path, '%s.tfrecords' % kind)
  if include_labels:
    labels_binary_path = os.path.join(path, '%s_labels.tfrecords' % kind)
    return data_binary_path, labels_binary_path
  return data_binary_path

def get_training_set(dataset_name, include_labels=True):
  return get_dataset(dataset_name, include_labels, 'train')

def get_test_set(dataset_name, include_labels=False):
  return get_dataset(dataset_name, include_labels, 'test')

def restore_logs(logfile):
  if tf.gfile.Exists(logfile):
    tf.gfile.DeleteRecursively(logfile)
  tf.gfile.MakeDirs(logfile)
