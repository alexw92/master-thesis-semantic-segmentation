import scipy.io as sio
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

label_colours = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 29], [219, 219, 0], [106, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [69, 129, 180], [219, 19, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 69]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 79, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle

osm_label_colours = [(0, 0, 0),        # 0 = unlabelled
                     (213, 131, 7),    # 0 = building
                     (0, 153, 0),      # 0 = wood
                     (0, 0, 204),      # 0 = water
                     (76, 0, 153),     # 0 = road
                     (255, 255, 102)]  # 0 = residential

kaggle_dstl_label_colours = [(0, 0, 0),    # 0 = OTHER
                              (213, 131, 7),  # 1 = BUILDING
                              (213, 140, 30),  # 2 = MIS_STRUCTURES
                              (76, 0, 153),  # 3 = ROAD
                              (80, 0, 160),  # 4 = TRACK
                              (0, 153, 0),  # 5 = TREES
                              (10, 100, 10),  # 6 = CROPS
                              (0, 0, 204),  # 7 = WATERWAY
                              (0, 0, 250),  # 8 = STANDING_WATER
                              (255, 255, 102),  # 9 = VEHICLE_LARGE
                              (245, 245, 110)]  # 10 = VEHICLE_SMALL

vaihingen_label_colours = [(76, 0, 153),  # Impervious_surfaces =
                           (213, 131, 7),  # Buildings
                           (5, 120, 5),  # vegetation
                           (0, 153, 0),  # Tree
                           (255, 255, 102),  # Car
                           (200, 0, 0)]  # Clutter


def get_coloredGT(image, num_classes,  dataset):
    """ store label data to colored image """
    # OSM Colors
    if num_classes == 150:
        color_table = read_labelcolours(matfn)
    elif num_classes == 6:
        color_table = osm_label_colours
    elif num_classes == 5:
        color_table = osm_label_colours[:5]
    elif num_classes == 11:
        color_table = kaggle_dstl_label_colours
    else:
        color_table = label_colours
    if 'vaihingen' in dataset:
        color_table = vaihingen_label_colours

    r = image.copy()
    g = image.copy()
    b = image.copy()
    for l in range(0, num_classes):
        r[image == l] = color_table[l][0]
        g[image == l] = color_table[l][1]
        b[image == l] = color_table[l][2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    return im


matfn = './utils/color150.mat'
def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, img_shape, num_classes, dataset):
    if num_classes == 150:
        color_table = read_labelcolours(matfn)
    elif num_classes == 6:
        color_table = osm_label_colours
    elif num_classes == 5:
        color_table = osm_label_colours[:5]
    elif num_classes == 11:
        color_table = kaggle_dstl_label_colours
    else:
        color_table = label_colours
    if 'vaihingen' in dataset:
        color_table = vaihingen_label_colours

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))
    
    return pred


def get_certainity(mask, img_shape):
    certainity_color = [(255,255,255)]
    color_mat = tf.constant(certainity_color, dtype=tf.float32)
    mask = tf.reshape(mask,(-1,1))
    certainity = tf.matmul(mask, color_mat)
    certainity = tf.reshape(certainity, (1, img_shape[0], img_shape[1], 3))

    return certainity


def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch
