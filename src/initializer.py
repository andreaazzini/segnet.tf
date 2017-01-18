import numpy as np

WEIGHT_FILE = './input/vgg16_weights.npz'

def initialize(params, sess):
  weights = np.load(WEIGHT_FILE)
  keys = sorted(weights.keys())[:26]
  for i, k in enumerate(keys):
    sess.run(params[i].assign(weights[k]))
