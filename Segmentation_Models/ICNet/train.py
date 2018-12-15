"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np
import math

from tqdm import trange
from model import ICNet_BN
from tools import decode_labels, prepare_label
from image_reader import ImageReader

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# If you want to apply to other datasets, change following four lines
# DATA_DIR = '../../ANN_DATA/GoogleImages_17_600x600_cropped'
# DATA_LIST_PATH = './list/osm_train_list.txt'
DATA_DIR = '/autofs/stud/werthmann/master/ANN_DATA/de_top15_cropped'
DATA_LIST_PATH = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_train_list.txt'
DATA_LIST_PATH_VAL = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_r_val_list.txt'
DATA_LIST_PATH_E1 = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_e1_list.txt'
DATA_LIST_PATH_E2 = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_e2_list.txt'
DATA_LIST_PATH_E3 = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_e3_list.txt'
DATA_LIST_PATH_E4 = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_e4_list.txt'
DATA_LIST_PATH_E5 = '/autofs/stud/werthmann/master/code/Segmentation_Models/ICNet/list/de_top15_e5_list.txt'
IGNORE_LABEL = 255 # The class number of background
INPUT_SIZE = '720, 720' # Input size for training

BATCH_SIZE = 16 
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 6
NUM_STEPS = 1000000
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = './master/code/Segmentation_Models/ICNet/model/cityscape/icnet_cityscapes_trainval_90k_bnnomerge.npy'
# './snapshots/' changed to execute the else part of the restore-if clause
SNAPSHOT_DIR = './master/code/Segmentation_Models/ICNet/mysnapshots'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 1000
CROSS_VAL = None
VALIDATE = True
PRINT_STEPS = True

weights_detop15 = tf.constant([0.975644, 1.025603, 0.601745, 6.600600, 1.328684, 0.454776])
weights_eutop25 = tf.constant([0.970664, 1.031165, 0.790741, 5.320133, 1.384649, 0.718765])
weights_world2k = tf.constant([0.879195, 1.439660, 0.683112, 4.628286, 1.159291, 0.322113])

# weights for datasets without residential area (= other)
weights_detop15_nores = tf.constant([0.303529, 1.000000, 0.604396, 5.941638, 1.305352])
weights_eutop25_nores = tf.constant([0.400486, 1.000000, 0.766842, 5.159342, 1.342801])
weights_world2k_nores = tf.constant([0.203351, 1.241845, 0.589249, 3.992340, 1.000000])

weights_kaggledstl = tf.constant([0.014317, 0.227888, 2.175962, 1.000000, 0.300450, 0.081639,
                                  0.046646, 1.740426, 8.405148, 749.202109, 73.475000])
weights_vaihingen = tf.constant([0.808506, 0.855016, 1.086051, 0.926584, 18.435326, 26.644663])
dataset_class_weights = None

# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
#LAMBDA1 = 0.16
#LAMBDA2 = 0.4
#LAMBDA3 = 1.0
LAMBDA1 = 0.4
LAMBDA2 = 0.6
LAMBDA3 = 1.0


def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--datadir", type=str, default=DATA_DIR,
                        help="Where to load data from.")
    parser.add_argument("--datalist-path", type=str, default=DATA_LIST_PATH,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--datalist-path-val", type=str, default=DATA_LIST_PATH_VAL,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--datalist_path_epoch1", type=str, default=DATA_LIST_PATH_E1,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--datalist_path_epoch2", type=str, default=DATA_LIST_PATH_E2,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--datalist_path_epoch3", type=str, default=DATA_LIST_PATH_E3,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--datalist_path_epoch4", type=str, default=DATA_LIST_PATH_E4,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--datalist_path_epoch5", type=str, default=DATA_LIST_PATH_E5,
                        help="Where find the datalist containing the filenames.")
    parser.add_argument("--max_runtime", type=float, default=5.0,
                        help="maximum run time (hours) of this job (cronjob shall restart it)")
    parser.add_argument("--validate", action="store_true", default=VALIDATE,
                        help="whether to perform validation")
    parser.add_argument("--patience", type=int, default=4,
                        help="The number of validations to wait for improvement before training is stopped")
    parser.add_argument("--reset-patience", action="store_true",
                        help="Whether to reset the possibly stored patience value.")
    parser.add_argument("--print_steps", action="store_true", default=PRINT_STEPS,
                        help="whether to perform validation (currently only 1 batch! fixme!)")
    parser.add_argument("--cross_val", type=int, default=CROSS_VAL, choices=[1, 2, 3, 4, 5],
                        help="The number of epoch used for validation for cross val, must be number between 1 and 5")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--weights_for_dataset", type=str, default=None,
                        help="Name of the dataset (used to select the weights)",
                        choices=['de_top15', 'eu_top25', 'world2k', 'de_top15_nores', 'eu_top25_nores',
                                 'world2k_nores', 'kaggle_dstl', 'vaihingen', None])

    return parser.parse_args()


def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
    varlist=[]
    var_value =[]
    reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return varlist#, var_value


def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
        except:
            print('Not found: '+tensor_name)
        full_var_list.append(tensor_aux)
    return full_var_list


def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices


def print_assign_vars(sess):
    """
    For Training with Validation:
    Copies the the weights of
    :param sess:
    :return:
    """
    for v in tf.global_variables():
        if "val" in v.name:
            n_name = v.name.split("/")
            f_name = "/".join(n_name[1:])
            for l in tf.trainable_variables():
                if f_name == l.name:
                    sess.run(v.assign(l))

# def create_loss(output, label, num_classes, ignore_label):
#     raw_pred = tf.reshape(output, [-1, num_classes])
#     label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
#     label = tf.reshape(label, [-1,])
#
#     indices = get_mask(label, num_classes, ignore_label)
#     gt = tf.cast(tf.gather(label, indices), tf.int32)
#     pred = tf.gather(raw_pred, indices)
#
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
#     reduced_loss = tf.reduce_mean(loss)
#
#     return reduced_loss


# TODO 1.) Test weighted loss approach
def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    # added class weights  un, bui, wo, wa, ro, res
    #class_weights = tf.constant([0.153, 0.144, 0.245, 0.022, 0.11, 0.325])

    #  class weight calculation used in segnet
    global dataset_class_weights
    if dataset_class_weights is None:
        dataset_class_weights = tf.constant([1 for i in range(num_classes)])
    class_weights = dataset_class_weights#tf.constant([0.975644, 1.025603, 0.601745, 6.600600, 1.328684, 0.454776])
    weights = tf.gather(class_weights, gt)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=gt, weights=weights)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss


def main():
    """Create the model and start the training."""
    args = get_arguments()
    print("SAVE TO "+args.snapshot_dir)
    datalists_epoch = {1: args.datalist_path_epoch1,
                       2: args.datalist_path_epoch2,
                       3: args.datalist_path_epoch3,
                       4: args.datalist_path_epoch4,
                       5: args.datalist_path_epoch5}
    if args.cross_val:
        val_epoch = int(args.cross_val)
        train_epochs = [1, 2, 3, 4, 5]
        train_epochs.remove(val_epoch)
        train_lists = [datalists_epoch[i] for i in train_epochs]
        val_lists = datalists_epoch[val_epoch]
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    max_runtime = args.max_runtime
    max_time_seconds = 3600 * max_runtime
    epochs_until_val = 3

    global dataset_class_weights
    if args.weights_for_dataset is None:
        dataset_class_weights = None
    elif args.weights_for_dataset == 'de_top15':
        dataset_class_weights = weights_detop15
    elif args.weights_for_dataset == 'eu_top25':
        dataset_class_weights = weights_eutop25
    elif args.weights_for_dataset == 'world2k':
        dataset_class_weights = weights_world2k
    elif args.weights_for_dataset == 'kaggle_dstl':
        dataset_class_weights = weights_kaggledstl
    elif args.weights_for_dataset == 'vaihingen':
        dataset_class_weights = weights_vaihingen
    elif args.weights_for_dataset == 'de_top15_nores':
        dataset_class_weights = weights_detop15_nores
    elif args.weights_for_dataset == 'eu_top25_nores':
        dataset_class_weights = weights_eutop25_nores
    elif args.weights_for_dataset == 'world2k_nores':
        dataset_class_weights = weights_world2k_nores


    coord = tf.train.Coordinator()

    if args.cross_val:
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                args.datadir,
                train_lists,
                input_size,
                args.random_scale,
                args.random_mirror,
                args.ignore_label,
                IMG_MEAN,
                coord)
            image_batch, label_batch = reader.dequeue(args.batch_size)

            # for validation
            reader_val = ImageReader(
                args.datadir,
                val_lists,
                input_size,
                args.random_scale,
                args.random_mirror,
                args.ignore_label,
                IMG_MEAN,
                coord)
            image_batch_val, label_batch_val = reader_val.dequeue(args.batch_size)
    else:

        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                args.datadir,
                args.datalist_path,
                input_size,
                args.random_scale,
                args.random_mirror,
                args.ignore_label,
                IMG_MEAN,
                coord)
            image_batch, label_batch = reader.dequeue(args.batch_size)


            # for validation
            reader_val = ImageReader(
                args.datadir,
                args.datalist_path_val,
                input_size,
                args.random_scale,
                args.random_mirror,
                args.ignore_label,
                IMG_MEAN,
                coord)
            image_batch_val, label_batch_val = reader_val.dequeue(args.batch_size)


    
    net = ICNet_BN({'data': image_batch}, is_training=True, num_classes=args.num_classes, filter_scale=args.filter_scale)
    with tf.variable_scope("val"):
        net_val = ICNet_BN({'data': image_batch_val}, is_training=True, num_classes=args.num_classes,
                           filter_scale=args.filter_scale)

    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']


    # early stop variables
    last_val_loss_tf = tf.Variable(10000.0, name="last_loss")
    steps_total_tf = tf.Variable(0, name="steps_total")
    val_increased_t_tf = tf.Variable(0, name="loss_increased_t")



    if args.not_restore_last:
        restore_var = [v for v in tf.global_variables() if 'conv6_cls' not in v.name and 'val' not in v.name
                       and 'sub4_out' not in v.name and 'sub24_out' not in v.name and 'sub124_out' not in v.name]
    else:
        # to load last layer, the line 78 in network.py has to be removed too and ignore_missing set to False
        # see https://github.com/hellochick/ICNet-tensorflow/issues/50 BCJuan
        # don't restore val vars
        restore_var = [v for v in tf.trainable_variables() if 'val' not in v.name]#tf.global_variables()
        # don't train val variables
    all_trainable = [v for v in tf.trainable_variables() if ((
        'beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma) and 'val' not in v.name]
        # all_trainable = [v for v in tf.trainable_variables() if
        #                  ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]


   # print([v for v in tf.global_variables() if v.name in["last_val_loss","steps_total","val_increased_t"]])
   # restore_var.extend([v for v in tf.global_variables() if v.name in["last_val_loss","steps_total","val_increased_t"]])

    # assert not np.any(np.isnan(sub4_out))
    loss_sub4 = create_loss(sub4_out, label_batch, args.num_classes, args.ignore_label)
    loss_sub24 = create_loss(sub24_out, label_batch, args.num_classes, args.ignore_label)
    loss_sub124 = create_loss(sub124_out, label_batch, args.num_classes, args.ignore_label)
   # l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                 ('weights' in v.name and 'val' not in v.name)]
    reduced_loss = LAMBDA1 * loss_sub4 +  LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    ####################### Loss Calculation FOR VALIDATION

    sub4_out_val = net_val.layers['sub4_out']
    sub24_out_val = net_val.layers['sub24_out']
    sub124_out_val = net_val.layers['conv6_cls']

    loss_sub4_val = create_loss(sub4_out_val, label_batch_val, args.num_classes, args.ignore_label)
    loss_sub24_val = create_loss(sub24_out_val, label_batch_val, args.num_classes, args.ignore_label)
    loss_sub124_val = create_loss(sub124_out_val, label_batch_val, args.num_classes, args.ignore_label)
    l2_losses_val = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                     ('weights' in v.name and 'val' in v.name)]

    reduced_loss_val = LAMBDA1 * loss_sub4_val + LAMBDA2 * loss_sub24_val + LAMBDA3 * loss_sub124_val + tf.add_n(
        l2_losses_val)
    ####################### End Loss Calculation FOR VALIDATION

    # Using Poly learning rate policy 
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)

    # start time
    glob_start_time = time.time()

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

    if '.npy' not in args.restore_from:
        ckpt = tf.train.get_checkpoint_state(args.restore_from)
    else:
        ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        vars_to_restore = get_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path)
        # print(vars_to_restore)
        # print([v.name for v in restore_var])
        # thanks to https://stackoverflow.com/a/50216949/8862202
        # v.name[:-2] to transform 'conv1_1_3x3_s2/weights:0' to 'conv1_1_3x3_s2/weights'
        vars_to_restore = [v for v in restore_var if 'val' not in v.name and  v.name[:-2] in vars_to_restore]
        # print(vars_to_restore)
        #loader = tf.train.Saver(var_list=restore_var)
        loader = tf.train.Saver(var_list=vars_to_restore)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('Restore from pre-trained model...')
        net.load(args.restore_from, sess)
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if args.reset_patience:
        z = tf.assign(val_increased_t_tf, 0)
        sess.run(z)

    print(sess.run(last_val_loss_tf))
    print(sess.run(steps_total_tf))
    print(sess.run(val_increased_t_tf))

    if not args.cross_val:
        val_epoch_len = len(reader_val.image_list)
        val_num_steps = val_epoch_len//args.batch_size
        # Iterate over training steps.
        last_val_loss = sess.run(last_val_loss_tf)
        val_increased_t = sess.run(val_increased_t_tf)
        best_model_step = 0
        total_steps = sess.run(steps_total_tf)
        for step in range(total_steps, args.num_steps+total_steps):
            start_time = time.time()
            feed_dict = {step_ph: step}
            if step % args.save_pred_every == 0:

                # validating
                if args.validate:
                    print("validating: ")
                    print_assign_vars(sess)
                    print("Assigned vars for validation. ")
                    loss_sum = 0
                    for val_step in trange(val_num_steps, desc='validation', leave=True):
                        loss_value_v, loss1_v, loss2_v, loss3_v = sess.run([reduced_loss_val, loss_sub4_val,
                                                                            loss_sub24_val, loss_sub124_val], feed_dict=feed_dict)
                        loss_sum = loss_sum + loss_value_v
                    loss_avg = loss_sum/val_num_steps

                    if loss_avg > last_val_loss:
                        val_increased_t = val_increased_t + 1
                        if val_increased_t >= args.patience:
                            print("Terminated Training, Best Model (at step %d) saved 4 validations ago"%best_model_step)
                            f = open("./FINISHED_ICNET", "w+")
                            f.close()
                            break

                    else:
                        val_increased_t = 0
                        best_model_step = step

                    print(
                        'VALIDATION COMPLETE step {:d}\tVal_Loss Increased {:d}/{:d} times\t total loss = {:.3f}'
                        ' last loss = {:.3f}'.format(
                            step, val_increased_t, args.patience, loss_avg, last_val_loss))

                    last_val_loss = loss_avg
                    steps_assign = tf.assign(steps_total_tf, step)
                    last_val_assign = tf.assign(last_val_loss_tf, last_val_loss)
                    increased_assign = tf.assign(val_increased_t_tf, val_increased_t)
                    print("loss avg "+str(loss_avg))
                    print(sess.run(steps_assign))
                    print(sess.run(last_val_assign))
                    print(sess.run(increased_assign))


                # Saving

                loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24,
                                                               loss_sub124, train_op], feed_dict=feed_dict)
                save(saver, sess, args.snapshot_dir, step)

                # check if max run time is already over
                elapsed = time.time() - glob_start_time
                if (elapsed + 300) > max_time_seconds:
                    print("Training stopped: max run time elapsed")
                    os.remove("./RUNNING_ICNET")
                    break
            else:
                loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24,
                                                               loss_sub124, train_op], feed_dict=feed_dict)
            duration = time.time() - start_time
            print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(step, loss_value, loss1, loss2, loss3, duration))
        train_duration = time.time() - glob_start_time
        print('Total training time: '+str(train_duration))
    else:
        # Training with cross validation
        print("Training-Mode CROSS VALIDATION")
        val_epoch_len = len(reader_val.image_list)
        val_num_steps = val_epoch_len//args.batch_size
        print("Val epoch length %d, Num steps %d"%(val_epoch_len, val_num_steps))
        last_val_loss = math.inf
        val_not_imp_t = 0

            # train

        for step in range(1000000):
            feed_dict = {step_ph: step}
            train_start = time.time()
            loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24,
                                                       loss_sub124, train_op], feed_dict=feed_dict)
            duration_t = time.time()-train_start
            if args.print_steps:
                print(
                    'step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(
                        step, loss_value, loss1, loss2, loss3, duration_t))

            if step % args.save_pred_every == 0:
                # save and validate
                # SAVE previously trained model
                save(saver, sess, args.snapshot_dir, step)
                # Validate
                print("validating: ")
                start_time = time.time()
                print_assign_vars(sess)
                print("Assigned vars for validation. ")
                loss_sum = 0
                for val_step in trange(val_num_steps, desc='validation', leave=True):
                    loss_value_v, loss1_v, loss2_v, loss3_v = sess.run([reduced_loss_val, loss_sub4_val,
                                                                        loss_sub24_val, loss_sub124_val], feed_dict=feed_dict)
                    loss_sum = loss_sum + loss_value_v
                duration = time.time()-start_time
                loss_avg = loss_sum/val_num_steps
                print(
                    'VALIDATION COMPLETE step {:d} \t total loss = {:.3f} \t duration = {:.3f}'.format(
                        step, loss_avg, duration))

            if loss_avg >= last_val_loss:
                 val_not_imp_t = val_not_imp_t + 1
                 if val_not_imp_t >= 4:
                     print("Terminated Training, Best Model saved 5 validations before")
                     f = open("./FINISHED_ICNET", "w+")
                     f.close()
                     break

            else:
                val_not_imp_t = 0

            last_val_loss = loss_avg

            # check if max run time is already over
            elapsed = time.time() - glob_start_time
            if (elapsed + 300) > max_time_seconds:
                print("Training stopped: max run time elapsed")
                os.remove("./RUNNING_ICNET")
                break

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    print("ICNET")
    main()
