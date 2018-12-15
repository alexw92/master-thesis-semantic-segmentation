# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

import math
import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
import my_metrics

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [608, 608],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


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


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  # vars_to_restore = get_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path)
  # print([v.name for v in tf.global_variables()])
  # vars_to_restore = [v for v in tf.global_variables() if v.name[:-2] in vars_to_restore]

  with tf.Graph().as_default():
    samples = input_generator.get(
        dataset,
        FLAGS.eval_crop_size,
        FLAGS.eval_batch_size,
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        dataset_split=FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.eval_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    for eval_scale in FLAGS.eval_scales:
      predictions_tag += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:
      predictions_tag += '_flipped'
    prediction_tag_2 = 'iou_per_class'
    prediction_tag_confMatrix = 'confusion_matrix'

    # segnet iou in numpy
    def fast_hist(gt, pred, n_clss):
        # true false mask where gt is valid
        # k = (gt >= 0) & (gt < n_clss)
        # return tf.reshape(tf.bincount(n_clss * tf.cast(gt[k], tf.int8) + pred[k], minlength=n_clss ** 2),
        #                   [n_clss, n_clss])
        print(type(gt))
        print(type(pred))
        return tf.reshape(tf.bincount(
            tf.add(
                tf.multiply(n_clss, gt),
                tf.cast(pred, tf.int32)),
            minlength=n_clss ** 2),
            [n_clss, n_clss])

    def get_hist(predictions, labels, num_class, batch_size):
        print(predictions.shape)
        # num_class = predictions.shape[3]
        # batch_size = predictions.shape[0]
        hist = tf.zeros((num_class, num_class), dtype=tf.int32)
        print(labels.shape)
        print(predictions.shape)
        for i in range(batch_size):
            # hist += fast_hist(labels[i], predictions[i], num_class)
            hist += fast_hist(labels[i], predictions[i], num_class)
        return hist

    # Define the evaluation metric.
    metric_map = {}
    metric_map[predictions_tag] = tf.metrics.mean_iou(
        predictions, labels, dataset.num_classes, weights=weights)
    
    # metric_map[prediction_tag_2] = tf.map_fn(get_hist, [predictions, labels,
    #                                         dataset.num_classes, FLAGS.eval_batch_size])
    # metric_map[prediction_tag_2] = get_hist(predictions, labels,
    #                                         num_class=dataset.num_classes, batch_size=FLAGS.eval_batch_size)

    # wieso geht das nicht?
    # metric_map[prediction_tag_confMatrix] = tf.contrib.metrics.confusion_matrix(labels, predictions,
    #                                     num_classes=dataset.num_classes)

   # metric_map['precision'] = tf.contrib.metrics.streaming_precision(predictions, labels)
    metric_map['accuracy'] = tf.metrics.accuracy(labels, predictions)
    # compute each class's iou thx 2 MrZhousf :D
    mean_iou_v, update_op = my_metrics.iou(predictions, labels, dataset.num_classes, weights=weights)
    acc_v, update_op2 = my_metrics.acc(predictions, labels, dataset.num_classes, weights=weights)
    for index in range(0, dataset.num_classes):
        metric_map[str(index)+'_' + segmentation_dataset.get_classname(FLAGS.dataset, index) + '_iou'] =\
            (mean_iou_v[index], update_op[index])
        metric_map[str(index)+'_' + segmentation_dataset.get_classname(FLAGS.dataset, index) + '_acc'] =\
            (acc_v[index], update_op2[index])

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))


    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(
          metric_value, metric_name, print_summary=True)

    num_batches = int(
        math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

    tf.logging.info('Eval num images %d', dataset.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.eval_batch_size, num_batches)

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
