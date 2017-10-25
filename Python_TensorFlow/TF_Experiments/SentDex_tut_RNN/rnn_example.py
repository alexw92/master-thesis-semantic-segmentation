# Error on new tensorflow version using old code from video:
# AttributeError: module 'tensorflow.python.ops.rnn' has no attribute 'rnn'
# Fix https://stackoverflow.com/a/42311633

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
# this saves in 'C:/tmp/data/'
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# takes like 2 min for calc with this config
hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes]), name="weights"),
             'biases': tf.Variable(tf.random_normal([n_classes]), name="biases")}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)  # old version: x = tf.split(0, n_chunks, x)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)),
                                          y: mnist.test.labels}))

        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)


# Might not work because lstm hasn't been saved explicitly
# https://stackoverflow.com/questions/40442098/saving-and-restoring-a-trained-lstm-in-tensor-flow
# confirms that it works though
def restore_rnn_model():
    tf.reset_default_graph()

    # Create some variables.
    v1 = tf.get_variable("weights", [rnn_size, n_classes])
    v2 = tf.get_variable("biases", [n_classes])
    # Add ops to save and restore only `v2` using the name "v2"
    saver = tf.train.Saver({"weights": v1})

    # Use the saver object normally after that.
    with tf.Session() as sess:
        # Initialize v2 since the saver will not.
        v2.initializer.run()
        saver.restore(sess, "/tmp/model.ckpt")

        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())
    return v1, v2


# train_neural_network(x)
v1, v2 = restore_rnn_model()



