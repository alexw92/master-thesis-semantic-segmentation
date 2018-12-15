
import numpy as np
import os
import sys
from PIL import Image
from tqdm import trange
import glob
from tqdm import tqdm
import tensorflow as tf



def mse(imageAF, imageBF):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    im_frame_a = Image.open(imageAF)
    imageA = np.array(im_frame_a.getdata())
    im_frame_b = Image.open(imageBF)
    imageB = np.array(im_frame_b.getdata())
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def mse_tf(sess, mse_tensor, imageAF, imageBF):
    im_frame_a = Image.open(imageAF)
    imageA = np.array(im_frame_a.getdata())
    im_frame_b = Image.open(imageBF)
    imageB = np.array(im_frame_b.getdata())
    return sess.run(mse_tensor, feed_dict={A: imageA, B: imageB})




def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()
    print(global_vars)
    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))




sample = sys.argv[1]
folder = sys.argv[2]
min = 1000000
res = ""

A = tf.placeholder(dtype=tf.float32, shape=[360000, 3])
B = tf.placeholder(dtype=tf.float32, shape=[360000, 3])
mse_tensor = tf.metrics.mean_squared_error(A, B)

init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    initialize_uninitialized_global_variables(sess)
    sess.run(init_l)

    for im_path in tqdm(glob.glob(folder+"/*.png")):
        err = mse_tf(sess, mse_tensor, sample, im_path)[1]
        if err < min:
            min = err
            res = im_path
            print(min)
            print(res+" "+sample)
    print(" most similar to %s with err= %d is %s" % (sample, min, res))

