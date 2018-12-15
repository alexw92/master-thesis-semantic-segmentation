
import matplotlib.pyplot as plt
import numpy as np
import cv2
from os.path import join
import os
import random
from functools import partial


from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, concatenate
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf

from keras.backend.tensorflow_backend import _to_tensor
from keras.backend.common import _EPSILON

import Map_Stuff.Map_Loader as maploader


K.set_image_dim_ordering('tf')


model_filepath = 'mymodel_5features_numep50.h5'
data_dirpath = '../ANN_DATA'
size = 224
n_cls = 2
batch_size = 1
num_images = 200  # total number of images used in one complete epoch
num_epoch = 50


class BatchGenerator:
    @staticmethod
    def to_one_hot(img, n_cls):
        return (np.arange(n_cls) == img[:, :, None]).astype(int)

    def __init__(self, txt_filepath, size, n_cls, batch_size):
        self.lines = []
        # for line in open(txt_filepath, 'r').readlines():
        #     line = line.strip()
        #     if len(line) > 0:
        #         self.lines.append(line)
        self.size = size
        self.n_cls = n_cls
        self.batch_size = batch_size
        self.i = 0

    def get_sample(self):
        if self.i == 0:
            random.shuffle(self.lines)
        # orig_filepath, gt_filepath = self.lines[self.i].split()
        # orig = cv2.imread(orig_filepath)  # 1 and 3 channels swapped
        # orig = cv2.resize(orig, (size, size))
        # gt = cv2.imread(gt_filepath)[:, :, 0]
        # gt = cv2.resize(gt, (self.size, self.size), cv2.INTER_NEAREST)
        # gt = BatchGenerator.to_one_hot(gt, self.n_cls + 1)  # + neutral class
        # self.i = (self.i + 1) % len(self.lines)
        # orig = np.zeros((self.size, self.size, 3))
        # gt = np.zeros((self.size, self.size, self.n_cls + 1))
        orig, gt = maploader.get_sample()
        gt = gt = BatchGenerator.to_one_hot(gt, self.n_cls + 1)
        return orig, gt

    def get_batch(self):
        while True:
            orig_batch = np.zeros((self.batch_size, self.size, self.size, 3))
            gt_batch = np.zeros((self.batch_size, self.size, self.size, self.n_cls + 1))
            for i in range(self.batch_size):
                orig, gt = self.get_sample()
                orig_batch[i] = orig
                gt_batch[i] = gt
            yield orig_batch, gt_batch

    def get_size(self):
        return num_images


def my_acc(target, output):
    target = K.cast(target, tf.int32)
    correct_count = K.sum(K.cast(K.equal(K.cast(K.argmax(target, axis=-1), tf.int32),
                                         K.cast(K.argmax(output, axis=-1), tf.int32)), tf.int32))
    neutral_count = K.sum(K.cast(K.equal(target[:, :, :, -1], K.variable(1, dtype=tf.int32)), tf.int32))
    total_count = K.prod(K.shape(output)[:-1]) - neutral_count
    return tf.cast(correct_count / total_count, tf.float32)


def get_acc(true, pred):
    total_count = np.prod(true.shape[:-1])
    total_count -= np.sum(true[:, :, :, -1] == 1)
    correct_count = np.sum(np.argmax(true, axis=3) == np.argmax(pred, axis=3))
    return correct_count / total_count


cls2col = {
    0: [0, 0, 0],       # classless - black
    1: [255, 255, 0],   # building  - yellow
    2: [0, 200, 0],     # wood      - green
    3: [0, 0, 200],     # water     - blue
    4: [50, 50, 50]     # road      - grey
}

class_weights = {
    0 : 1.,
    1: 50.,
    2: 0
}


def pred2featuremap(img_pred):
    img_pred = np.argmax(img_pred, axis=2)
    img_pred = img_pred.astype(np.uint8)
    h, w = img_pred.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in cls2col.items():
        result[img_pred == k] = v
    return result


def my_bce(target, output):
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)

    return -tf.reduce_sum(target[:, :, :, :-1] * tf.log(output),
                          axis=-1)


train_batch_generator = BatchGenerator(join(data_dirpath, 'train.txt'), size, n_cls, batch_size)
val_batch_generator = BatchGenerator(join(data_dirpath, 'val.txt'), size, n_cls, batch_size)

# load model
model = load_model(model_filepath, custom_objects={'my_acc': my_acc, 'my_bce': my_bce})

# plot sample
for x, y in val_batch_generator.get_batch():
    pred = model.predict(x)
    featuremap = pred2featuremap(pred[0, :, :, :])
    print('Acc: %.3f' % get_acc(y, pred))
    img = x[0, :, :, :]  # get first image of batch
    fig, ax = plt.subplots(1, 2) # plot image and prediction
    ax[0].imshow(featuremap)
    ax[1].imshow(img.astype(np.uint8))
    plt.show()