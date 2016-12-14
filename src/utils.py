import os
import tensorflow as tf

_colors = {
  'cookeat': [
    [255, 245, 0],   # coke
    [3, 102, 255],   # lemon
    [220, 0, 255],   # pear
    [255, 7, 201],   # banana
    [0, 68, 112],    # table
    [255, 255, 255], # background
    [0, 0, 0]        # void
  ],
  'segnet-13': [
    [0, 128, 192],
    [128, 0, 0],
    [64, 0, 128],
    [192, 192, 128],
    [64, 64, 128],
    [60, 40, 222],
    [64, 64, 0],
    [128, 64, 128],
    [255, 69, 0],
    [192, 128, 128],
    [128, 128, 128],
    [128, 128, 0],
    [0, 0, 0]
  ],
  'segnet-32': [
    [64, 128, 64],   # Animal
    [192, 0, 128],   # Archway
    [0, 128, 192],   # Bicyclist
    [0, 128, 64],    # Bridge
    [128, 0, 0],     # Building
    [64, 0, 128],    # Car
    [64, 0, 192],    # CartLuggagePram
    [192, 128, 64],  # Child
    [192, 192, 128], # Column_Pole
    [64, 64, 128],   # Fence
    [128, 0, 192],   # LaneMkgsDriv
    [192, 0, 64],    # LaneMkgsNonDriv
    [128, 128, 64],  # Misc_Text
    [192, 0, 192],   # MotorcycleScooter
    [128, 64, 64],   # OtherMoving
    [64, 192, 128],  # ParkingBlock
    [64, 64, 0],     # Pedestrian
    [128, 64, 128],  # Road
    [128, 128, 192], # RoadShoulder
    [0, 0, 192],     # Sidewalk
    [192, 128, 128], # SignSymbol
    [128, 128, 128], # Sky
    [64, 128, 192],  # SUVPickupTruck
    [0, 0, 64],      # TrafficCone
    [0, 64, 64],     # TrafficLight
    [192, 64, 128],  # Train
    [128, 128, 0],   # Tree
    [192, 128, 192], # Truck_Bus
    [64, 0, 64],     # Tunnel
    [192, 192, 0],   # VegetationMisc
    [0, 0, 0],       # Void
    [64, 192, 0]     # Wall
  ],
  'single-coke': [
    [46, 201, 252],  # Coke
    [255, 255, 255], # Background
  ]
}

def _get_dataset(dataset_name, include_labels, kind):
  path = os.path.join('input', dataset_name)
  data_binary_path = os.path.join(path, '%s.tfrecords' % kind)
  if include_labels:
    labels_binary_path = os.path.join(path, '%s_labels.tfrecords' % kind)
    return data_binary_path, labels_binary_path
  return data_binary_path

def get_training_set(dataset_name, include_labels=True):
  return _get_dataset(dataset_name, include_labels, 'train')

def get_test_set(dataset_name, include_labels=False):
  return _get_dataset(dataset_name, include_labels, 'test')

def colors_of_dataset(dataset_name):
  return _colors[dataset_name]

def restore_logs(logfile):
  if tf.gfile.Exists(logfile):
    tf.gfile.DeleteRecursively(logfile)
  tf.gfile.MakeDirs(logfile)
