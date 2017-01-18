import classifier
import convnet as cnn
import tensorflow as tf

class SegNetAutoencoder:
  def __init__(self, n, strided=False, max_images=3):
    self.max_images = max_images
    self.n = n
    self.strided = strided

  def conv(self, x, channels_shape, name):
    return cnn.conv(x, [3, 3], channels_shape, 1, name)

  def conv2(self, x, channels_shape, name):
    return cnn.conv(x, [3, 3], channels_shape, 2, name)

  def deconv(self, x, channels_shape, name):
    return cnn.deconv(x, [3, 3], channels_shape, 1, name)

  def deconv2(self, x, channels_shape, name):
    return cnn.deconv(x, [3, 3], channels_shape, 2, name)

  def pool(self, x):
    return cnn.max_pool(x, 2, 2)

  def unpool(self, x):
    out = tf.concat_v2([x, tf.zeros_like(x)], 3)
    out = tf.concat_v2([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
      out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
      return tf.reshape(out, out_size)
    else:
      shv = tf.shape(x)
      ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
      ret.set_shape([None, None, None, sh[3]])
    return ret

  def inference_with_pooling(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv(conv1, [64, 64], 'conv1_2')
      pool1 = self.pool(conv2)

    with tf.variable_scope('pool2'):
      conv3 = self.conv(pool1, [64, 128], 'conv2_1')
      conv4 = self.conv(conv3, [128, 128], 'conv2_2')
      pool2 = self.pool(conv4)

    with tf.variable_scope('pool3'):
      conv5 = self.conv(pool2, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv(conv6, [256, 256], 'conv3_3')
      pool3 = self.pool(conv7)

    with tf.variable_scope('pool4'):
      conv8 = self.conv(pool3, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv(conv9, [512, 512], 'conv4_3')
      pool4 = self.pool(conv10)

    with tf.variable_scope('pool5'):
      conv11 = self.conv(pool4, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv(conv12, [512, 512], 'conv5_3')
      pool5 = self.pool(conv13)

    with tf.variable_scope('unpool1'):
      unpool1 = self.unpool(pool5)
      deconv1 = self.deconv(unpool1, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      unpool2 = self.unpool(deconv3)
      deconv4 = self.deconv(unpool2, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      unpool3 = self.unpool(deconv6)
      deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      unpool4 = self.unpool(deconv9)
      deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      unpool5 = self.unpool(deconv11)
      deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    rgb_image = classifier.rgb(deconv13)
    # rgb_image = tf.Print(rgb_image, [rgb_image], summarize=30)
    # mask = tf.reshape(tf.sign(tf.reduce_max(rgb_image, 3)), [-1, 224, 224, 1])
    # mask = tf.Print(mask, [mask], summarize=30)
    # rgba_image = tf.concat(3, [rgb_image, mask]) * 255
    tf.summary.image('output', rgb_image, max_outputs=self.max_images)

    return deconv13

  def strided_inference(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv2(conv1, [64, 64], 'conv1_2')

    with tf.variable_scope('pool2'):
      conv3 = self.conv(pool1, [64, 128], 'conv2_1')
      conv4 = self.conv2(conv3, [128, 128], 'conv2_2')

    with tf.variable_scope('pool3'):
      conv5 = self.conv(pool2, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv2(conv6, [256, 256], 'conv3_3')

    with tf.variable_scope('pool4'):
      conv8 = self.conv(pool3, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv2(conv9, [512, 512], 'conv4_3')

    with tf.variable_scope('pool5'):
      conv11 = self.conv(pool4, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv2(conv12, [512, 512], 'conv5_3')

    with tf.variable_scope('unpool1'):
      deconv1 = self.deconv2(unpool1, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      deconv4 = self.deconv2(unpool2, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      deconv7 = self.deconv2(unpool3, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      deconv10 = self.deconv2(unpool4, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      deconv12 = self.deconv2(unpool5, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    rgb_image = classifier.rgb(deconv13)
    # rgb_image = tf.Print(rgb_image, [rgb_image], summarize=30)
    # mask = tf.reshape(tf.sign(tf.reduce_max(rgb_image, 3)), [-1, 224, 224, 1])
    # mask = tf.Print(mask, [mask], summarize=30)
    # rgba_image = tf.concat(3, [rgb_image, mask]) * 255
    tf.summary.image('output', rgb_image, max_outputs=self.max_images)

    return deconv13

  def inference(self, images):
    if self.strided:
      return self.strided_inference(images)
    return self.inference_with_pooling(images)
