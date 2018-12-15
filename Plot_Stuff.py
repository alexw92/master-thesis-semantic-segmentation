import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np
import re
import sys
import time

"""
Different learning rate policies

learning rate policies are as follows (source caffe):
    - fixed: always return base_lr.
    - step: return base_lr * gamma ^ (floor(iter / step))
    - exp: return base_lr * gamma ^ iter
    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
    - multistep: similar to step but it allows non uniform steps defined by
      stepvalue
    - poly: the effective learning rate follows a polynomial decay, to be
      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
    - sigmoid: the effective learning rate follows a sigmod decay
      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))

 where base_lr, max_iter, gamma, step, stepvalue and power are defined
 in the solver parameter protocol buffer, and iter is the current iteration.
"""


def plot_lr_fixed():
    pass


def plot_lr_step():
    pass


def plot_lr_exp():
    pass


def plot_lr_inv():
    pass


def plot_lr_multistep():
    pass


def plot_lr_poly(base_lr=1e-3, power=0.9, max_iter=100000):
    t = np.arange(0, max_iter, 10)
    pow(base_lr*(1 - t / max_iter), power)
    plt.figure()
    plt.plot(t, pow(base_lr*(1 - t / max_iter), power), 'g')

    plt.xlabel('Number of iterations')
    plt.ylabel('Learning rate')
    plt.title('Learning rate policy \'Poly\'')
    plt.show()


def plot_lr_sigmoid():
    pass


def plot_loss_from_slurm(slurm_file='C:/Uni/Masterstudium/ma-werthmann/code/train_logs_slurm/ICNet/slurm_icnet_25000_only_gamma.out'
                         , rmWindow=100, modelname='icnet'):
    """

    :param slurm_file: The slurm-123456.out file produced by slurm with contains lines of loss
    :return: plots the loss
    psp_regex = 'step\s([\d]*)\s\/t\sloss\s=\s([\d]*\.[\d]*),'
    """
    if 'psp' in slurm_file or 'psp' in modelname:
        model_name = 'PSPNet'
        regex = 'step\s([\d]*)\s\/t\sloss\s=\s([\d]*\.[\d]*),'
    elif 'icnet' in slurm_file or 'icnet' in modelname:
        model_name = 'ICNet'
        regex = 'step\s([\d]*)\s\\t\stotal\sloss\s=\s([\d]*\.[\d]*),'
    elif 'segnet' in slurm_file or 'segnet' in modelname:
        model_name = 'SegNet'
        regex = 'step\s([\d]*),\sloss\s=\s([\d]*\.[\d]*)'
    steps = []
    losses = []
    with open(slurm_file, 'rt', encoding="utf-8") as file:
        line = 'lol'
        while line != '':
            line = file.readline()
            m = re.search(regex, line)
            if m is not None:
                step = int(m.group(1))
                loss = float(m.group(2))
                steps.append(step)
                losses.append(loss)
    steps = np.asarray(steps)
    losses = np.asarray(losses)

    # This calculates the running mean by using a convolution window of rmWindow
    running_mean = np.convolve(losses, np.ones((rmWindow,)) / rmWindow, mode='valid')
    print(len(running_mean))

    plt.figure()
    plt.ylim(0, 2.5)
    plt.plot(steps, losses, 'g', linewidth=0.2)
    plt.plot(steps[:-rmWindow+1], running_mean, 'r')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Vaihingen without OSM pretraining')#model_name+' - Learning Curve -
    plt.show()


# Todo convert to numpy
# def decode_labels(mask, img_shape=(600,600), num_classes=6):
#     osm_label_colours = [(0, 0, 0),  # 0 = unlabelled
#                          (213, 131, 7),  # 0 = building
#                          (0, 153, 0),  # 0 = wood
#                          (0, 0, 204),  # 0 = water
#                          (76, 0, 153),  # 0 = road
#                          (255, 255, 102)]  # 0 = residential
#     color_table = osm_label_colours
#
#     color_mat = color_table#= tf.constant(color_table, dtype=tf.float32)
#     onehot_output = tf.one_hot(mask, depth=num_classes)
#     onehot_output = tf.reshape(onehot_output, (-1, num_classes))
#     pred = tf.matmul(onehot_output, color_mat)
#     pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))
#
#     return pred


if __name__ == '__main__':

    if len(sys.argv) == 2:
        plot_loss_from_slurm(sys.argv[1])
    elif len(sys.argv) == 3:
        plot_loss_from_slurm(sys.argv[1], int(sys.argv[2]))
    else:
      #  plot_lr_poly()
       # plot_loss_from_slurm()
        plot_loss_from_slurm(slurm_file='C:/Users/Alex/Desktop/icnet_vaihingen_lr=e-3.out')
