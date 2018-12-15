from __future__ import print_function
import argparse
import os
import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from model import ICNet, ICNet_BN
from image_reader import read_labeled_image_list

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# define setting & model configuration
ADE20k_param = {'name': 'ade20k',
                'input_size': [480, 480],
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'data_dir': '../../ADEChallengeData2016/', 
                'data_list': './list/ade20k_val_list.txt'}
                
cityscapes_param = {'name': 'cityscapes',
                    'input_size': [1025, 2049],
                    'num_classes': 19,
                    'ignore_label': 255,
                    'data_dir': '/data/cityscapes_dataset/cityscape', 
                    'data_list': './list/cityscapes_val_list.txt'}

de_top15_train_param = {'name': 'de_top15_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/de_top15_cropped',
                    'data_list': './list/de_top15_r_train_list.txt'}

de_top15_test_param = {'name': 'de_top15_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/de_top15_cropped',
                    'data_list': './list/de_top15_r_test_list.txt'}

eu_top25_train_param = {'name': 'eu_top25_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/eu_top25_cropped',
                    'data_list': './list/eu_top25_r_train_list.txt'}

eu_top25_test_param = {'name': 'eu_top25_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/eu_top25_cropped',
                    'data_list': './list/eu_top25_r_test_list.txt'}

de_top15_nores_train_param = {'name': 'de_top15_nores_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 5,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/de_top15_cropped',
                    'data_list': './list/nores_de_top15_r_train_list.txt'}

de_top15_nores_test_param = {'name': 'de_top15_nores_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 5,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/de_top15_cropped',
                    'data_list': './list/nores_de_top15_r_test_list.txt'}

world2k_train_param = {'name': 'world2k_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/world_tiny2k_cropped',
                    'data_list': './list/world_tiny2k_r_train_list.txt'}

world2k_test_param = {'name': 'world2k_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/world_tiny2k_cropped',
                    'data_list': './list/world_tiny2k_r_test_list.txt'}

world2k_nores_train_param = {'name': 'world2k_nores_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 5,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/world_tiny2k_cropped',
                    'data_list': './list/nores_world_tiny2k_r_train_list.txt'}

world2k_nores_test_param = {'name': 'world2k_nores_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 5,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/world_tiny2k_cropped',
                    'data_list': './list/nores_world_tiny2k_r_test_list.txt'}

eu_top25_nores_train_param = {'name': 'eu_top25_nores_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 5,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/eu_top25_cropped',
                    'data_list': './list/nores_eu_top25_r_train_list.txt'}

eu_top25_nores_test_param = {'name': 'eu_top25_nores_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 5,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/FINAL_DATASETS/eu_top25_cropped',
                    'data_list': './list/nores_eu_top25_r_test_list.txt'}

kaggle_dstl_test_param = {'name': 'kaggle_dstl_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 11,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/DSTL_Challenge_Dataset',
                    'data_list': './list/kaggle_dstl_test_list.txt'}

kaggle_dstl_train_param = {'name': 'kaggle_dstl_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 11,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/DSTL_Challenge_Dataset',
                    'data_list': './list/kaggle_dstl_train_list.txt'}

kaggle_dstl_val_param = {'name': 'kaggle_dstl_val',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 11,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/DSTL_Challenge_Dataset',
                    'data_list': './list/kaggle_dstl_val_list.txt'}

vaihingen_test_param = {'name': 'vaihingen_test',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/vaihingen',
                    'data_list': './list/vaihingen_test_list.txt'}

vaihingen_train_param = {'name': 'vaihingen_train',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/vaihingen',
                    'data_list': './list/vaihingen_train_list.txt'}

vaihingen_val_param = {'name': 'vaihingen_val',
                    'input_size': [608, 608], # width must be <= target-offset (for preprocessing)
                    'num_classes': 6,
                    'ignore_label': 255,
                    'data_dir': 'G:/Datasets/vaihingen',
                    'data_list': './list/vaihingen_val_list.txt'}

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy', 
               'trainval': './model/icnet_cityscapes_trainval_90k.npy',
               'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
               'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
               # 'others': './model/mymodel/'}
               'others': 'C:/Users/Alex/Desktop/113450_eigenegewichte'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--measure-time", action="store_true",
                        help="whether to measure inference time")
    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Will only be evaluated if --model=others, Use this parameter for your custom model path")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'de_top15_test', 'de_top15_train',
                                 'eu_top25_test', 'eu_top25_train', 'kaggle_dstl_train', 'kaggle_dstl_test',
                                 'kaggle_dstl_val', 'vaihingen_train', 'vaihingen_test',
                                 'vaihingen_val', 'de_top15_nores_test', 'de_top15_nores_train',
                                 'eu_top25_nores_test', 'eu_top25_nores_train',
                                 'world2k_nores_test', 'world2k_nores_train'],
                        required=True)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--data_list", type=str, default=None,
                        help="The datalist containing a list of filename tuples."
                             " Should be used together with --data_dir")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="The directory containing the dataset files")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Max steps for evaluation.")

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

time_list = []
def calculate_time(sess, net, pred, feed_dict):
    start = time.time()
    sess.run(net.layers['data'], feed_dict=feed_dict)
    data_time = time.time() - start

    start = time.time()
    sess.run(pred, feed_dict=feed_dict)
    total_time = time.time() - start

    inference_time = total_time - data_time

    time_list.append(inference_time)
    print('average inference time: {}'.format(np.mean(time_list)))

def preprocess(img, param):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    shape = param['input_size']

    if param['name'] == 'cityscapes':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
    elif param['name'] == 'de_top15_train':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
    elif param['name'] == 'de_top15_test':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
    elif param['name'] == 'ade20k':
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_bilinear(img, shape, align_corners=True)
    elif param['name'] == 'kaggle_dstl_train':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
    else:
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
        
    return img

def main():
    args = get_arguments()
    
    if args.dataset == 'ade20k':
        param = ADE20k_param
    elif args.dataset == 'cityscapes':
        param = cityscapes_param
    elif args.dataset == 'de_top15_train':
        param = de_top15_train_param
    elif args.dataset == 'de_top15_test':
        param = de_top15_test_param
    elif args.dataset == 'world2k_test':
        param = world2k_test_param
    elif args.dataset == 'world2k_train':
        param = world2k_train_param
    elif args.dataset == 'eu_top25_train':
        param = eu_top25_train_param
    elif args.dataset == 'eu_top25_test':
        param = eu_top25_test_param
    elif args.dataset == 'de_top15_nores_test':
        param = de_top15_nores_test_param
    elif args.dataset == 'de_top15_nores_train':
        param = de_top15_nores_train_param
    elif args.dataset == 'eu_top25_nores_train':
        param = eu_top25_nores_train_param
    elif args.dataset == 'eu_top25_nores_test':
        param = eu_top25_nores_test_param
    elif args.dataset == 'world2k_nores_test':
        param = world2k_nores_test_param
    elif args.dataset == 'world2k_nores_train':
        param = world2k_nores_train_param
    elif args.dataset == 'kaggle_dstl_train':
        param = kaggle_dstl_train_param
    elif args.dataset == 'kaggle_dstl_test':
        param = kaggle_dstl_test_param
    elif args.dataset == 'kaggle_dstl_val':
        param = kaggle_dstl_val_param
    elif args.dataset == 'vaihingen_train':
        param = vaihingen_train_param
    elif args.dataset == 'vaihingen_test':
        param = vaihingen_test_param
    elif args.dataset == 'vaihingen_val':
        param = vaihingen_val_param
    # Set placeholder
    image_filename = tf.placeholder(dtype=tf.string)
    anno_filename = tf.placeholder(dtype=tf.string)

    # Read & Decode image
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    ori_shape = tf.shape(img)
    img = preprocess(img, param)

    model = model_config[args.model]
    net = model({'data': img}, num_classes=param['num_classes'], 
                    filter_scale=args.filter_scale, evaluation=True)

    # Predictions.
    raw_output = net.layers['conv6_cls']

    raw_output_up = tf.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    raw_pred = tf.expand_dims(raw_output_up, dim=3)

    # mIoU
    pred_flatten = tf.reshape(raw_pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])

    mask = tf.not_equal(raw_gt, param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if args.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes']+1)
    elif args.dataset == 'cityscapes':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'osm':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'de_top15_train':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'de_top15_test':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'eu_top25_train':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'eu_top25_test':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'kaggle_dstl_train':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'kaggle_dstl_test':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'vaihingen_train':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'vaihingen_test':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    else:
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    
    sess.run(init)
    sess.run(local_init)

    model_path = model_paths[args.model]
    custom_model_path = args.model_path
    if args.model == 'others':
        if not custom_model_path: # if no custom model_path was given use default path of model
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                loader = tf.train.Saver(var_list=tf.global_variables())
                load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
                load(loader, sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found.')
        else:
            ckpt = tf.train.get_checkpoint_state(custom_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                loader = tf.train.Saver(var_list=tf.global_variables())
                load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
                load(loader, sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found.')
    else:
        net.load(model_path, sess)
        print('Restore from {}'.format(model_path))

    data_list = param['data_list']
    data_dir = param['data_dir']
    if args.data_list:
        data_list = args.data_list
    if args.data_dir:
        data_dir = args.data_dir
    img_files, anno_files = read_labeled_image_list(data_dir, data_list)
    steps = min(len(img_files), args.max_steps)
    hist = np.zeros((param['num_classes'], param['num_classes']))
    eval_file = open(args.dataset, 'wt+')
    for i in trange(steps, desc='evaluation', leave=True):
        feed_dict = {image_filename: img_files[i], anno_filename: anno_files[i]}
        gt_out, pred_out, _ = sess.run([gt, pred, update_op], feed_dict=feed_dict)

        lhist = get_hist(gt_out, pred_out, num_class=param['num_classes'], batch_size=1)
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

        if i > 0 and args.measure_time:
            calculate_time(sess, net, raw_pred, feed_dict)
    eval_file.close()
    print('mIoU: {}'.format(sess.run(mIoU)))
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))
    cl_name = get_dataset_classnames(param['name'])
    for cls in range(hist.shape[0]):
        iou = iu[cls]
        if float(hist.sum(1)[cls]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[cls] / float(hist.sum(1)[cls])
        print("    class %s accuracy = %f, IoU =  %f" % (cl_name[cls].ljust(12), acc, iou))
   

def fast_hist(gt, pred, n_clss):
    # true false mask where gt is valid
    k = (gt >= 0) & (gt < n_clss)
    return np.bincount(n_clss * gt[k].astype(int) + pred[k], minlength=n_clss ** 2).reshape(n_clss, n_clss)


def get_hist(predictions, labels, num_class, batch_size):
    #num_class = predictions.shape[3]

    # batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels, predictions, num_class)
    return hist



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

if __name__ == '__main__':
    main()
