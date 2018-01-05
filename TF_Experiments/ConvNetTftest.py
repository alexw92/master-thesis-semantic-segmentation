# from datasets import dataset_utils
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from tensorflow.contrib.slim.python.slim.data.dataset import Dataset


# 7m0s
# https://mediaspace.gatech.edu/media/TF-SlimA+A+Lightweight+Library+for+Defining%2C+Training+and+Evaluating+Complex+Models+in+TensorFlow+-+Nathan+Silberman/1_00mrferp
def MyModel(images, num_classes, is_training = False):
    net = slim.conv2d(images, 32, [5, 5], padding='SAME', scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2d')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
    net = slim.fully_connected(net, num_classes,
                               activation_fn=None, scope='fc4')
    return net

