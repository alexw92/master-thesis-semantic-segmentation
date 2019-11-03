import os
import tensorflow as tf
flags = tf.app.flags

FLAGS = flags.FLAGS

# 1
# Create SavedModelBuilder class
# defines where the model will be exported
export_path_base = FLAGS.export_model_dir
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(FLAGS.model_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# 2

tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
tensor_info_height = tf.saved_model.utils.build_tensor_info(image_height_tensor)
tensor_info_width = tf.saved_model.utils.build_tensor_info(image_width_tensor)

# output tensor info
tensor_info_output = tf.saved_model.utils.build_tensor_info(predictions_tf)

# Defines the DeepLab signatures, uses the TF Predict API
# It receives an image and its dimensions and output the segmentation mask
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_input, 'height': tensor_info_height, 'width': tensor_info_width},
        outputs={'segmentation_map': tensor_info_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

# 3
with tf.Session as sess:
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        })

# export the model
builder.save(as_text=True)
print('Done exporting!')