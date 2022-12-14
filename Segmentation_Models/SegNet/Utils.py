import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PIL import Image

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

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/device:CPU:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(
        name,
        shape,
        initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_certainity(mask, img_shape, num_classes=6):
    certainity_color = [(255,255,255)]
    color_mat = tf.constant(certainity_color, dtype=tf.float32)
    #
    mask = tf.reduce_max(mask, axis=-1)
    onehot_output = tf.reshape(mask, (-1, 1))

    #mask = tf.reshape(mask,(-1,1))
    certainity = tf.matmul(onehot_output, color_mat)
    certainity = tf.reshape(certainity, (1, img_shape[0], img_shape[1], 3))

    return certainity

def writeImage(image, filename, dataset='osm'):
    """ store label data to colored image """
    # OSM Colors"detop15_nores", "eutop25_nores", "worldtiny2k_nores"
    if dataset == 'osm':
        Unlabelled = [0, 0, 0]
        Building = [213, 131, 7]
        Wood = [0, 153, 0]
        Water = [0, 0, 204]
        Road = [76, 0, 153]
        Residential = [255, 255, 102]
        label_colours = np.array([Unlabelled, Building, Wood, Water, Road, Residential])
    elif dataset == 'vaihingen':
        Impervious_surfaces = [76, 0, 153]
        Buildings = [213, 131, 7]
        vegetation = [5, 120, 5]
        Tree = [0, 153, 0]
        Car = [255, 255, 102]
        Clutter = [200, 0, 0]
        label_colours = np.array([Impervious_surfaces,
        Buildings, vegetation, Tree,
        Car, Clutter])
    else:
        Unlabelled = [0, 0, 0]
        Building = [213, 131, 7]
        Wood = [0, 153, 0]
        Water = [0, 0, 204]
        Road = [76, 0, 153]
        Residential = [255, 255, 102]
        label_colours = np.array([Unlabelled, Building, Wood, Water, Road, Residential])
    # CamVid Colors
    # Sky = [128,128,128]
    # Building = [128,0,0]
    # Pole = [192,192,128]
    # Road_marking = [255,69,0]
    # Road = [128,64,128]
    # Pavement = [60,40,222]
    # Tree = [128,128,0]
    # SignSymbol = [192,128,128]
    # Fence = [64,64,128]
    # Car = [64,0,128]
    # Pedestrian = [64,64,0]
    # Bicyclist = [0,128,192]
    # Unlabelled = [0,0,0]

    r = image.copy()
    g = image.copy()
    b = image.copy()
    for l in range(0, 6):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)


def storeImageQueue(data, labels, step):
    """ data and labels are all numpy arrays """
    for i in range(len(data)):
        index = 0
        im = data[i]
        la = labels[i]
        im = Image.fromarray(np.uint8(im))
        im.save("batch_im_s%d_%d.png" % (step, i))
        writeImage(np.reshape(la, (600, 600)), "batch_la_s%d_%d.png" % (step, i))


def fast_hist(gt, pred, n_clss):
    # true false mask where gt is valid
    k = (gt >= 0) & (gt < n_clss)
    return np.bincount(n_clss * gt[k].astype(int) + pred[k], minlength=n_clss ** 2).reshape(n_clss, n_clss)


def get_hist(predictions, labels):
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


def print_hist_summery(hist, dataset=None):
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    num_class = hist.shape[0]
    if num_class==11:
        print("class shape last hist summary "+str(num_class))
        cl_name = kaggle_class_names
    elif dataset== 'vaihingen':
        cl_name = vaihingen_class_names
    elif 'nores' in dataset:
        cl_name = class_name
    else:
        cl_name = class_name
    for cls in range(hist.shape[0]):
        iou = iu[cls]
        if float(hist.sum(1)[cls]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[cls] / float(hist.sum(1)[cls])
        print("    class %s accuracy = %f, IoU =  %f" % (cl_name[cls].ljust(12), acc, iou))


def per_class_acc(predictions, label_tensor, dataset=None):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    if num_class==11:
        print("class shape last hist summary "+str(num_class))
        cl_name = kaggle_class_names
    elif dataset== 'vaihingen':
        cl_name = vaihingen_class_names
    elif 'nores' in dataset:
        cl_name = class_name
    else:
        cl_name = class_name
    for cls in range(num_class):
        iou = iu[cls]
        if float(hist.sum(1)[cls]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[cls] / float(hist.sum(1)[cls])
        print("    class %s accuracy = %f, IoU =  %f" % (cl_name[cls].ljust(12),acc, iou))