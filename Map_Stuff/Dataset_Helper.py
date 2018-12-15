import glob
import math
import os.path
import re
import sys
import tensorflow as tf
import shutil

"""
PART 1/2 TO TRANSFORM OSM DATA FOR DEEPLABv3+
Continue with 

Given the following structure
    + ANN_DATA
        + osm_dataset1
            + gt_data
                - GT_area_lon1xlat1.png
                - GT_area_lon2xlat2.png
                - ...
            - area_lon1xlat1.png
            - area_lon2xlat2.png

this Script will create  

    + ANN_DATA
        + osm_dataset1
            + deeplab_osm_dataset1
                + satImage
                    + train
                    + test
                    + val
                + gt
                    + train
                    + test
                    + val
            + gt_data
                - GT_area_lon1xlat1.png
                - GT_area_lon2xlat2.png
                - ...
            - area_lon1xlat1.png
            - area_lon2xlat2.png         
            
"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('osm_dataset',
                           '../ANN_DATA/world_tiny2k_cropped',
                           'OSM dataset root folder.')
tf.app.flags.DEFINE_string('list_train',
                           './list/train.txt',
                           'list of train data')
tf.app.flags.DEFINE_string('list_test',
                           './list/test.txt',
                           'list of test data')
tf.app.flags.DEFINE_string('list_val',
                           './list/val.txt',
                           'list of ')


def _read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")

        image = os.path.join(data_dir, image)
        mask = os.path.join(data_dir, mask)
        mask = mask.strip()
        if not tf.gfile.Exists(image):
            print("missing image "+image)
            continue
           # raise ValueError('Failed to find file: ' + image)

        if not tf.gfile.Exists(mask):
            print("missing mask " + mask)
            continue
            # raise ValueError('Failed to find file: ' + mask)

        images.append(image)
        masks.append(mask)

    return images, masks


def main(unused_argv):
    # Create dirs
    basename = os.path.basename(FLAGS.osm_dataset)
    dirname = os.path.dirname(FLAGS.osm_dataset)
    new_dataset_dir = os.path.join(dirname, basename, "deeplab_" + basename)

    new_list_train = os.path.join(new_dataset_dir, "deeplab_" + os.path.basename(FLAGS.list_train))
    new_list_test = os.path.join(new_dataset_dir, "deeplab_" + os.path.basename(FLAGS.list_test))
    new_list_val = os.path.join(new_dataset_dir, "deeplab_" + os.path.basename(FLAGS.list_val))

    if not os.path.exists(new_dataset_dir):
        os.makedirs(new_dataset_dir)

    long_dir_name = new_dataset_dir
    new_dataset_dir = os.path.basename(new_dataset_dir)
    if not os.path.exists(os.path.join(new_dataset_dir, 'satImage', 'train')):
        os.makedirs(os.path.join(new_dataset_dir, 'satImage', 'train'))
    if not os.path.exists(os.path.join(new_dataset_dir, 'gt', 'train')):
        os.makedirs(os.path.join(new_dataset_dir, 'gt', 'train'))
    if not os.path.exists(os.path.join(new_dataset_dir, 'satImage', 'test')):
        os.makedirs(os.path.join(new_dataset_dir, 'satImage', 'test'))
    if not os.path.exists(os.path.join(new_dataset_dir, 'gt', 'test')):
        os.makedirs(os.path.join(new_dataset_dir, 'gt', 'test'))
    if not os.path.exists(os.path.join(new_dataset_dir, 'satImage', 'val')):
        os.makedirs(os.path.join(new_dataset_dir, 'satImage', 'val'))
    if not os.path.exists(os.path.join(new_dataset_dir, 'gt', 'val')):
        os.makedirs(os.path.join(new_dataset_dir, 'gt', 'val'))

    for list in [FLAGS.list_train, FLAGS.list_test, FLAGS.list_val]:
        if list == FLAGS.list_train:
            dir_sat = os.path.join(new_dataset_dir, 'satImage', 'train')
            dir_gt = os.path.join(new_dataset_dir, 'gt', 'train')
            new_list = new_list_train
        if list == FLAGS.list_test:
            dir_sat = os.path.join(new_dataset_dir, 'satImage', 'test')
            dir_gt = os.path.join(new_dataset_dir, 'gt', 'test')
            new_list = new_list_test
        if list == FLAGS.list_val:
            dir_sat = os.path.join(new_dataset_dir, 'satImage', 'val')
            dir_gt = os.path.join(new_dataset_dir, 'gt', 'val')
            new_list = new_list_val
        image_names, mask_names = _read_labeled_image_list(FLAGS.osm_dataset, list)
        # move to new dir structure
        print("copying data to "+dir_sat)
        for image in image_names:
            shutil.copy2(image, dir_sat)
        for label in mask_names:
            shutil.copy2(label, dir_gt)

        with open(new_list, 'wt') as f:
            for x, y in zip(image_names, mask_names):
                x = os.path.join(dir_sat, os.path.basename(x))
                y = os.path.join(dir_gt, os.path.basename(y))
                f.write(x + ' ' + y + '\n')
            f.flush()


if __name__ == '__main__':
    tf.app.run()
