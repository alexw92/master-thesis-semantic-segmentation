'''
Feed forward network:

input layer> weights > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation

feed forward + backpropagation = epoch (epic^^)
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# 10 classes, 0-9
'''
"One hot", rest off
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
...
'''

# can be variable
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10    # number of outputs
batch_size = 100  # batch_size images at one time is fed to the network

# height x width
x = tf.placeholder('float', [None, 784])  # 28*28 pixels, no need to keep the initial shape of the image!
y = tf.placeholder('float')


def neural_network_model(data):

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    # W1*x1 + b1 = y1
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    # W2*y1 + b2 = y2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    # W3*y2 + b3 = y3
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    # OW*y3 + ob = o
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # difference between prediction and true solution
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    #                       learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # = cycles of feed forward + backpropagation
    hm_epochs = 10

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        # (10000, 784)
        # (10000, 10)
        print(mnist.test.images.shape)
        print(mnist.test.labels.shape)
        # Training
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):  # _ shorthand for  variable we don't care about
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sesh.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        # Testing
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
