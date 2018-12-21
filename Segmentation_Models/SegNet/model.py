import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from math import ceil
from tqdm import trange, tqdm
from scipy import misc

from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.ops import gen_nn_ops
# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, \
    _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage, get_certainity, predToLabelledImg
from Inputs import *

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
EVAL_BATCH_SIZE = 5
BATCH_SIZE = 5

NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 700
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist = []
    var_value = []
    reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return varlist  # , var_value


def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
        except:
            print('Not found: ' + tensor_name)
        full_var_list.append(tensor_aux)
    return full_var_list


def loss(logits, labels):
    """
        loss func without re-weighting
    """
    # Calculate the average cross entropy loss across the batch.
    logits = tf.reshape(logits, (-1, NUM_CLASSES))
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


def cal_loss(num_class, logits, labels, use_weights):
    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=num_class, head=use_weights)


def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    # shape = [filter_height, filter_width, in_channels, out_channels]
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


def get_deconv_filter(f_shape):
    """
      reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv


def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope + "_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope + "_bn",
                                                        reuse=True))


def inference(num_class, images, labels, batch_size, phase_train, use_weights):
    # print("input")
    # print(images.get_shape())
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                      name='norm1')
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
    # print("conv1")
    # print(conv1.get_shape())
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1')
    # print("pool1")
    # print(pool1.get_shape())
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
    # print("conv2")
    # print(conv2.get_shape())

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # print("pool2")
    # print(pool2.get_shape())
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")
    # print("conv3")
    # print(conv3.get_shape())
    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # print("pool3")
    # print(pool3.get_shape())
    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

    # print("conv4")
    # print(conv4.get_shape())
    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # print("pool4")
    # print(pool4.get_shape())
    """ End of encoder """
    """ start upsample """
    """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
"""
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 75, 75, 64], 2, "up4")
    # print("upsample4")
    # print(upsample4.get_shape())
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")
    # print("conv_decode4")
    # print(conv_decode4.get_shape())
    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3 = deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 150, 150, 64], 2, "up3")
    # print("upsample3")
    # print(upsample3.get_shape())
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")
    # print("conv_decode3")
    # print(conv_decode3.get_shape())

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2 = deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 300, 300, 64], 2, "up2")
    # print("upsample2")
    # print(upsample2.get_shape())
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")
    # print("conv_decode2")
    # print(conv_decode2.get_shape())

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1 = deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 600, 600, 64], 2, "up1")
    # print("upsample1")
    # print(upsample1.get_shape())

    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    # print("conv_decode1")
    # print(conv_decode1.get_shape())
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, num_class],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0005)
        conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [num_class], tf.constant_initializer(0.0))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    # print("logit")
    # print(logit.get_shape())
    if labels is None:
        return logit

    loss = cal_loss(num_class, conv_classifier, labels, use_weights=use_weights)

    return loss, logit


def train(total_loss, global_step, learning_rate):
    """ fix lr """
    lr = learning_rate
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    trainable = [v for v in tf.trainable_variables()
                 if v.name[:-2] not in ["last_loss", "steps_total", "loss_increased_t"]]
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    variables_averages_op = variable_averages.apply(trainable)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def infere(FLAGS):
    test_dir = FLAGS.test_dir  # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
    test_ckpt = FLAGS.testing
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    datadir = FLAGS.datadir
    outdir = FLAGS.output_dir
    max_infere = FLAGS.max_infere
    dataset = FLAGS.dataset
    batch_size = 1
    use_weights = FLAGS.use_weights

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # select weights
    # class weight calculation used in segnet
    # weights for dataset de_top14
    # "detop15", "eutop25", "worldtiny2k", "kaggle_dstl", "vaihingen",
    #                            "detop15_nores", "eutop25_nores", "worldtiny2k_nores"
    print('dont use weights during inference')
    use_weights = tf.constant([1.0 for i in range(FLAGS.num_class)])

    # feed only sat images to the net
    logits = inference(FLAGS.num_class, test_data_node, None, batch_size, phase_train, use_weights)
    # calc certainity grey scale
    certainity = get_certainity(logits, [image_h, image_w])

    pred = tf.argmax(logits, axis=3)
    # get moving avg
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY)
    # variables_to_restore = variable_averages.variables_to_restore()
    ckpt = tf.train.get_checkpoint_state(test_ckpt)
    vars_to_restore = get_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path)
    #  print(vars_to_restore)
    #  print([v.name for v in tf.global_variables()])
    vars_to_restore = [v for v in tf.global_variables()
                       if (not 'conv_classifier' in v.name or not FLAGS.not_restore_last)
                       and v.name[:-2] in vars_to_restore]
    # loader = tf.train.Saver(variables_to_restore)
    loader = tf.train.Saver(vars_to_restore)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)

    with tf.Session() as sess:
        # Load checkpoint
        loader.restore(sess, ckpt.model_checkpoint_path)

        images, labels = get_all_test_data(image_filenames, label_filenames, datadir, max_infere)

        hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
        # create out dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Warning! No Softmax Output before Argmax in Segnet! Certainity does not display probability! \n"
              "(Neither does a ConvNet with Softmax Output by the way ^^)")
        for i in trange(0, min(len(images), max_infere), desc="Infering", leave=True):
            image_fname = image_filenames[i]
            image = images[i]
            label = labels[i]  # not used currently, but will later be needed to save gt data along with pred
            print(image[0].shape)
            print(label[0].shape)
            feed_dict = {
                test_data_node: image,
                phase_train: False
            }

            dense_prediction, im, cert_img = sess.run([logits, pred, certainity], feed_dict=feed_dict)
            # output_image to verify
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            save_dir_name = os.path.dirname(outdir + '/' + image_fname)
            base_name = os.path.basename(outdir + '/' + image_fname)
            subdir = os.path.join(save_dir_name, str(i))
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            print(subdir)

            # writeImage(image, subdir + '/' + 'sat' + base_name)

            # deprec function TODO replace imsave with imageio.imwrite!
            misc.imsave(subdir + '/' + 'cert_' + base_name, cert_img[0])
            misc.imsave(subdir + '/' + 'sat_' + base_name, image[0].astype(int))
            writeImage(im[0], subdir + '/' + 'pred_' + base_name, dataset=dataset)
            writeImage(np.squeeze(label[0]), subdir + '/' + 'gt_' + base_name, dataset=dataset)
            # writeImage(im[0], 'out_image/'+str(image_filenames[count]).split('/')[-1])


def test(FLAGS):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    train_dir = FLAGS.log_dir  # /tmp3/first350/TensorFlow/Logs
    test_dir = FLAGS.test_dir  # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
    test_ckpt = FLAGS.testing
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    datadir = FLAGS.datadir
    outdir = FLAGS.output_dir
    dataset = FLAGS.dataset
    use_weights = FLAGS.use_weights
    # testing should set BATCH_SIZE = 1
    batch_size = 1

    # select weights
    # class weight calculation used in segnet
    # weights for dataset de_top14
    # "detop15", "eutop25", "worldtiny2k", "kaggle_dstl", "vaihingen",
    #                            "detop15_nores", "eutop25_nores", "worldtiny2k_nores"
    if not use_weights:
        print('dont use weights during evaluation')
    use_weights = tf.constant([1.0 for i in range(FLAGS.num_class)])

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

    test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, logits = inference(FLAGS.num_class, test_data_node, test_labels_node, batch_size, phase_train, use_weights)

    pred = tf.argmax(logits, axis=3)
    # get moving avg
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(test_ckpt)
        # Load checkpoint
        vars_to_restore = get_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path)
        #  print(vars_to_restore)
        #  print([v.name for v in tf.global_variables()])
        vars_to_restore = [v for v in tf.global_variables()
                           if (not 'conv_classifier' in v.name or not FLAGS.not_restore_last)
                           and v.name[:-2] in vars_to_restore]

        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(vars_to_restore)

        saver.restore(sess, ckpt.model_checkpoint_path)
        images, labels = get_all_test_data(image_filenames, label_filenames, datadir)
        threads = tf.train.start_queue_runners(sess=sess)
        hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
        eval_file = open(FLAGS.dataset, 'wt+')
        for image_batch, label_batch in tqdm(zip(images, labels), desc="Testing", leave=True, total=len(images)):

            feed_dict = {
                test_data_node: image_batch,
                test_labels_node: label_batch,
                phase_train: False
            }

            dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
            # output_image to verify
            if (FLAGS.save_image):
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                writeImage(im[0], outdir + '/testing_image.png', dataset=dataset)
                # writeImage(im[0], 'out_image/'+str(image_filenames[count]).split('/')[-1])
            # calc iou and acc values for each image (for significance test)
            lhist = get_hist(dense_prediction, label_batch)
            lacc = np.diag(lhist).sum() / lhist.sum()
            liu = np.diag(lhist) / (lhist.sum(1) + lhist.sum(0) - np.diag(lhist))
            acc_line = "acc"
            iou_line = "iou"
            for cls in range(lhist.shape[0]):
                iou = liu[cls]
                if float(lhist.sum(1)[cls]) == 0:
                    local_acc = 0.0
                else:
                    local_acc = np.diag(lhist)[cls] / float(lhist.sum(1)[cls])
                acc_line = acc_line + ", " + str(local_acc)
                iou_line = iou_line + ", " + str(iou)

            eval_file.write(acc_line + "\n")
            eval_file.write(iou_line + "\n")
            eval_file.flush()
            hist += lhist
            # count+=1
        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        eval_file.close()
        print("acc: ", acc_total)
        print("mean IU: ", np.nanmean(iu))
        cl_name = get_dataset_classnames(dataset)
        for cls in range(hist.shape[0]):
            iou = iu[cls]
            if float(hist.sum(1)[cls]) == 0:
                acc = 0.0
            else:
                acc = np.diag(hist)[cls] / float(hist.sum(1)[cls])
            print("    class %s accuracy = %f, IoU =  %f" % (cl_name[cls].ljust(12), acc, iou))


def get_dataset_classnames(dataset):
    class_name = [
        'Unlabelled',
        'Building',
        'Wood',
        'Water',
        'Road',
        'Residential'
    ]

    kaggle_class_names = [
        'OTHER', 'BUILDING', 'MIS_STRUCTURES', 'ROAD', 'TRACK',
        'TREES', 'CROPS', 'WATERWAY', 'STANDING_WATER',
        'VEHICLE_LARGE', 'VEHICLE_SMALL'
    ]

    vaihingen_class_names = ['Impervious_surfaces',
                             'Buildings', 'Low vegetation', 'Tree',
                             'Car', 'Clutter']
    if 'vaihingen' in dataset:
        return vaihingen_class_names
    elif 'kaggle' in dataset:
        return kaggle_class_names
    else:
        return class_name


def training(FLAGS, is_finetune=False):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    train_dir = FLAGS.log_dir  # /tmp3/first350/TensorFlow/Logs
    train_list = FLAGS.train_dir  # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
    val_dir = FLAGS.val_dir  # /tmp3/first350/SegNet-Tutorial/CamVid/val.txt
    finetune_ckpt = FLAGS.finetune
    not_restore_last = FLAGS.not_restore_last
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    datadir = FLAGS.datadir
    gpu_frac = FLAGS.gpu_usage
    dataset = FLAGS.dataset
    lr = FLAGS.learning_rate
    max_runtime = FLAGS.max_runtime
    use_weights = FLAGS.use_weights
    print("DEBUG 558: " + str(use_weights))

    max_time_seconds = 3600 * max_runtime
    TEST_ITER = int(get_dataset_params(dataset)["num_val"] / batch_size)
    EPOCH_ITER = int(get_dataset_params(dataset)["num_train"] / batch_size)
    EPOCHS_UNTIL_VAL = 1
    PATIENCE = FLAGS.patience
    print(TEST_ITER)
    print('batchsize training: ' + str(batch_size))
    print('max_steps training: ' + str(max_steps))
    print('lr training: ' + str(lr))
    print('Epochs until save ' + str(EPOCHS_UNTIL_VAL))

    # should be changed if your model stored by different convention
    startstep = 0 if not is_finetune else 0  # int((FLAGS.finetune.split('-')[-1]).split('.')[0])

    image_filenames, label_filenames = get_filename_list(train_list)
    val_image_filenames, val_label_filenames = get_filename_list(val_dir)

    with tf.Graph().as_default():

        train_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])

        train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        global_step = tf.Variable(0, trainable=False)

        # select weights
        # class weight calculation used in segnet
        # weights for dataset de_top14
        # "detop15", "eutop25", "worldtiny2k", "kaggle_dstl", "vaihingen",
        #                            "detop15_nores", "eutop25_nores", "worldtiny2k_nores"
        if not use_weights:
            print('dont use weights')
            use_weights = tf.constant([1.0 for i in range(FLAGS.num_class)])
        else:
            print('use weights')

            if dataset == "detop15":
                use_weights = np.array([0.975644, 1.025603, 0.601745, 6.600600, 1.328684, 0.454776])
            elif dataset == "eutop25":
                use_weights = np.array([0.970664, 1.031165, 0.790741, 5.320133, 1.384649, 0.718765])
            elif dataset == "worldtiny2k":
                use_weights = np.array([0.879195, 1.439660, 0.683112, 4.628286, 1.159291, 0.322113])
            elif dataset == "eutop25_nores":
                use_weights = np.array([0.400486, 1.000000, 0.766842, 5.159342, 1.342801])
            elif dataset == "detop15_nores":
                use_weights = np.array([0.303529, 1.000000, 0.604396, 5.941638, 1.305352])
            elif dataset == "worldtiny2k_nores":
                use_weights = np.array([0.203351, 1.241845, 0.589249, 3.992340, 1.000000])
            elif dataset == "vaihingen":
                use_weights = np.array([0.808506, 0.855016, 1.086051, 0.926584, 18.435326, 26.644663])
            elif dataset == "kaggle_dstl":
                use_weights = np.array([0.014317, 0.227888, 2.175962, 1.000000, 0.300450, 0.081639,
                                        0.046646, 1.740426, 8.405148, 749.202109, 73.475000])
            else:
                print('Error: No weights for dataset ' + dataset + ' could be found.')

        # early stop variables
        last_val_loss_tf = tf.Variable(10000.0, name="last_loss")
        steps_total_tf = tf.Variable(0, name="steps_total")
        val_increased_t_tf = tf.Variable(0, name="loss_increased_t")

        # For Inputs
        images, labels = OSMInputs(image_filenames, label_filenames, batch_size, datadir, dataset)

        val_images, val_labels = OSMInputs(val_image_filenames, val_label_filenames, batch_size, datadir, dataset)

        # Build a Graph that computes the logits predictions from the inference model.
        loss, eval_prediction = inference(FLAGS.num_class, train_data_node, train_labels_node, batch_size, phase_train,
                                          use_weights)
        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = train(loss, global_step, lr)

        # print(vars_to_restore)
        # print([v.name for v in restore_var])
        # thanks to https://stackoverflow.com/a/50216949/8862202
        # v.name[:-2] to transform 'conv1_1_3x3_s2/weights:0' to 'conv1_1_3x3_s2/weights'
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        summary_op = tf.summary.merge_all()

        glob_start_time = time.time()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        with tf.Session() as sess:
            # init always (necessary for early stop vars)
            init = tf.global_variables_initializer()
            sess.run(init)

            # Build an initialization operation to run below.
            if is_finetune:
                ckpt = tf.train.get_checkpoint_state(finetune_ckpt)
                ckpt_path = ckpt.model_checkpoint_path

                # restore only vars previously (in the last save) defined
                vars_to_restore = get_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path)
                #  print(vars_to_restore)
                #  print([v.name for v in tf.global_variables()])
                vars_to_restore = [v for v in tf.global_variables()
                                   if (not 'conv_classifier' in v.name or not FLAGS.not_restore_last)
                                   and v.name[:-2] in vars_to_restore]
                #  print(vars_to_restore)
                loader = tf.train.Saver(vars_to_restore, max_to_keep=10)

                # saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
                loader.restore(sess, ckpt_path)
                # saver.restore(sess, finetune_ckpt )
                print("Restored model parameters from {}".format(ckpt_path))
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            # debug, check early stop vars
            print(sess.run(last_val_loss_tf))
            print(sess.run(steps_total_tf))
            print(sess.run(val_increased_t_tf))

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Summery placeholders
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            average_pl = tf.placeholder(tf.float32)
            acc_pl = tf.placeholder(tf.float32)
            iu_pl = tf.placeholder(tf.float32)
            average_summary = tf.summary.scalar("test_average_loss", average_pl)
            acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
            iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

            last_val_loss = sess.run(last_val_loss_tf)
            val_not_imp_t = sess.run(val_increased_t_tf)
            total_steps = sess.run(steps_total_tf)
            for step in trange(startstep + total_steps, startstep + max_steps + total_steps, desc='training',
                               leave=True):
                image_batch, label_batch = sess.run([images, labels])
                # since we still use mini-batches in validation, still set bn-layer phase_train = True
                feed_dict = {
                    train_data_node: image_batch,
                    train_labels_node: label_batch,
                    phase_train: True
                }
                start_time = time.time()

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time

                # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 100 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    # time update
                    elapsed = time.time() - glob_start_time
                    remaining = max_time_seconds - elapsed

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch), %.1f seconds until stop')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch, remaining))

                    # eval current training batch pre-class accuracy
                    pred = sess.run(eval_prediction, feed_dict=feed_dict)
                    per_class_acc(pred, label_batch, dataset=dataset)
                    # generate image and send it to event file
                    argmax_t = tf.argmax(pred, axis=3)
                    argmax = sess.run(argmax_t)
                    sat = image_batch[0][:][:][:]
                    sat = np.expand_dims(sat, axis=0)
                    im = predToLabelledImg(argmax[0])
                    gt = predToLabelledImg(label_batch[0])
                    # concat images to a single 4D vector to get all images in single line in tensorboard
                    sat_pred_gt = np.concatenate((sat, im, gt),0)
                    sat_pred_gt_summary = tf.summary.image('sat_pred_gt', tf.convert_to_tensor(sat_pred_gt))
                    sat_pred_gt_summary = sess.run(sat_pred_gt_summary)
                    # img_sum_pred = tf.summary.image('pred img', tf.convert_to_tensor(im))
                    # img_sum_pred = sess.run(img_sum_pred)
                    # img_sum_gt = tf.summary.image('gt img', tf.convert_to_tensor(gt))
                    # img_sum_sat = tf.summary.image('satellite img', tf.convert_to_tensor(sat))
                    # img_sum_gt = sess.run(img_sum_gt)
                    # img_sum_sat = sess.run(img_sum_sat)
                    # debug line
                    # writeImage(argmax[0], str(step)+"_labelled.png", "osm")
                    #summary_writer.add_summary(img_sum_pred, step)
                   # summary_writer.add_summary(img_sum_gt, step)
                    summary_writer.add_summary(sat_pred_gt_summary, step)

                if step % EPOCH_ITER * EPOCHS_UNTIL_VAL == 0:
                    print("start validating.....")
                    total_val_loss = 0.0
                    hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
                    for test_step in range(int(TEST_ITER)):
                        val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

                        _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                            train_data_node: val_images_batch,
                            train_labels_node: val_labels_batch,
                            phase_train: True
                        })
                        total_val_loss += _val_loss
                        hist += get_hist(_val_pred, val_labels_batch)
                    total_val_loss = total_val_loss / TEST_ITER
                    print("val loss: {:.3f} , last val loss: {:.3f}".format(total_val_loss, last_val_loss))
                    acc_total = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss})
                    acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                    iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                    print_hist_summery(hist, dataset=dataset)
                    print(" end validating.... ")

                    if total_val_loss > last_val_loss:
                        val_not_imp_t = val_not_imp_t + 1
                        if val_not_imp_t >= PATIENCE:
                            print("Terminated Training, Best Model (at step %d) saved %d validations ago" % (
                            best_model_step, PATIENCE))
                            f = open("./FINISHED_SEGNET", "w+")
                            f.close()
                            break

                    else:
                        val_not_imp_t = 0
                        best_model_step = step
                    print("Loss not since improved %d times" % val_not_imp_t)
                    last_val_loss = total_val_loss

                    # update early stop tensors
                    steps_assign = tf.assign(steps_total_tf, step)
                    last_val_assign = tf.assign(last_val_loss_tf, last_val_loss)
                    increased_assign = tf.assign(val_increased_t_tf, val_not_imp_t)
                    print(sess.run(steps_assign))
                    print(sess.run(last_val_assign))
                    print(sess.run(increased_assign))

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(test_summary_str, step)
                    summary_writer.add_summary(acc_summary_str, step)
                    summary_writer.add_summary(iu_summary_str, step)
                    # Save the model checkpoint periodically.
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print("Checkpoint created at %s" % checkpoint_path)
                    # check if max run time is already over
                    elapsed = time.time() - glob_start_time
                    if (elapsed + 300) > max_time_seconds:
                        print("Training stopped: max run time elapsed")
                        os.remove("./RUNNING_SEGNET")
                        break

            coord.request_stop()
            coord.join(threads)
