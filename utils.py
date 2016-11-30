import tensorflow as tf

def restore_logs(logfile):
  if tf.gfile.Exists(logfile):
    tf.gfile.DeleteRecursively(logfile)
  tf.gfile.MakeDirs(logfile)
