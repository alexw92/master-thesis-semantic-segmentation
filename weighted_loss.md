# Weighted Loss Implementations

- [UNet Weighted Loss](https://stackoverflow.com/a/43679674) and [Orig Source](http://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/unet.html)

- Für Keras

```
def loss(y_true, y_pred):
	epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
	y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
	return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))
```

- Tensorflow Loss function with weights (built in ICNet train.py)

```
def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    # added class weights  un, bui, wo, wa, ro, res
    class_weights = tf.constant([0.153, 0.144, 0.245, 0.022, 0.11, 0.325])
    weights = tf.gather(class_weights, gt)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=gt, weights=weights)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss
```
