import tensorflow as tf

colors = [
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
]

colors = tf.cast(tf.pack(colors), tf.float32) / 255

def color_mask(tensor, color):
  return tf.reduce_all(tf.equal(tensor, color), 3)

def one_hot(labels):
  color_tensors = tf.unstack(colors)
  channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
  one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
  return one_hot_labels

def rgb(logits):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, 3)
  one_hot = tf.one_hot(argmax, 32, dtype=tf.float32)
  one_hot_matrix = tf.reshape(one_hot, [-1, 32])
  rgb_matrix = tf.matmul(one_hot_matrix, colors)
  rgb_tensor = tf.reshape(rgb_matrix, [-1, 224, 224, 3])
  return tf.cast(rgb_tensor, tf.float32)
