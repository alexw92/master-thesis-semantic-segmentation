import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import skimage
import skimage.io

# IMAGE_HEIGHT = 360
# IMAGE_WIDTH = 480
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 600
IMAGE_DEPTH = 3

NUM_CLASSES = 6


osm_topde15 = {
    "name": "osm_topde14",
    "num_train": 5626,
    "num_test": 703,
    "num_val": 703
}

osm_worldtiny2k = {
    "name": "osm_worldtiny2k",
    "num_train": 1472,
    "num_test": 184,
    "num_val": 184
}

osm_eutop25 =  {
    "name": "osm_eutop25",
    "num_train": 11921,
    "num_test": 1490,
    "num_val": 1490,
}

osm_topde15_nores = {
    "name": "detop15_nores",
    "num_train": 5626,
    "num_test": 703,
    "num_val": 703
}

osm_worldtiny2k_nores = {
    "name": "worldtiny2k_nores",
    "num_train": 1472,
    "num_test": 184,
    "num_val": 184
}

osm_eutop25_nores =  {
    "name": "eutop25_nores",
    "num_train": 11921,
    "num_test": 1490,
    "num_val": 1490,
}



kaggle_dstl =  {
    "name": "kaggle_dstl",
    "num_train": 480,
    "num_test": 60,
    "num_val": 60,
}

vaihingen =  {
    "name": "vaihingen",
    "num_train": 264,
    "num_test": 15,
    "num_val": 15,
}

def get_dataset_params(dataset):
    if("detop15" in dataset):
        return osm_topde15
    if("worldtiny2k" in dataset):
        return osm_worldtiny2k
    if "eutop25" in dataset:
        return osm_eutop25
    if "dstl" in dataset:
        return kaggle_dstl
    if "vaihingen" in dataset:
        return vaihingen
    if "detop15_nores" in dataset:
        return osm_topde15_nores
    if "eutop25_nores" in dataset:
        return osm_eutop25_nores
    if "worldtiny2k_nores" in dataset:
        return osm_worldtiny2k_nores

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 3-D Tensor of [height, width, 1] type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.image_summary('images', images)

    return images, label_batch


def CamVid_reader_seq(filename_queue, seq_length):
    image_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[0])
    label_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[1])
    image_seq = []
    label_seq = []
    for im, la in zip(image_seq_filenames, label_seq_filenames):
        print(im)
        imageValue = tf.read_file(tf.squeeze(im))
        labelValue = tf.read_file(tf.squeeze(la))
        image_bytes = tf.image.decode_png(imageValue)
        label_bytes = tf.image.decode_png(labelValue)
        image = tf.cast(tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.float32)
        label = tf.cast(tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1)), tf.int64)
        image_seq.append(image)
        label_seq.append(label)
    return image_seq, label_seq


def CamVid_reader(filename_queue, datadir):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    image_filename = filename_queue[0]
    label_filename = filename_queue[1]

    imageValue = tf.read_file(datadir + '/' + image_filename)
    labelValue = tf.read_file(datadir + '/' + label_filename)

    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)

    image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    return image, label


def get_filename_list(path):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    return image_filenames, label_filenames


def OSMInputs(image_filenames, label_filenames, batch_size, datadir, dataset):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    if 'detop15' in dataset:
        data = osm_topde15
    elif 'worldtiny2k' in dataset:
        data = osm_worldtiny2k
    elif 'eutop25' in dataset:
        data = osm_eutop25
    elif 'worldtiny2k_nores' in dataset:
        data = osm_worldtiny2k_nores
    elif 'detop15_nores' in dataset:
        data = osm_topde15_nores
    elif 'eutop25_nores' in dataset:
        data = osm_eutop25_nores
    elif 'dstl' in dataset:
        data = kaggle_dstl
    elif 'vaihingen' in dataset:
        data = vaihingen
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
    image, label = CamVid_reader(filename_queue, datadir)
    reshaped_image = tf.cast(image, tf.float32)

    min_fraction_of_examples_in_queue = 0.04
    min_queue_examples = int(data["num_train"] *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d OSM images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def get_all_test_data(im_list, la_list, datadir, max=None):
    images = []
    labels = []
    index = 0
    if max is not None:
        images = images[:max]
        labels = labels[:max]
    for im_filename, la_filename in zip(im_list, la_list):
        index = index + 1
        im = np.array(skimage.io.imread(datadir + "/" + im_filename), np.float32)
        im = im[np.newaxis]
        la = skimage.io.imread(datadir + "/" + la_filename)
        la = la[np.newaxis]
        la = la[..., np.newaxis]
        images.append(im)
        labels.append(la)
    return images, labels
