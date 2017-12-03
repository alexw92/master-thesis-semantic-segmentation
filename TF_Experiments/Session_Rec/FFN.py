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
import traindata as train
import eval as eval

# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

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
n_nodes_hl1 = 2000
n_nodes_hl2 = 2000
n_nodes_hl3 = 5000

n_classes = 52739    # number of outputs
batch_size = 10  # batch_size images at one time is fed to the network
n_csize = 52739
epoch_size = 500
# height x width
x = tf.placeholder('float', [None, n_csize])  # 28*28 pixels, no need to keep the initial shape of the image!
y = tf.placeholder('float', )


def neural_network_model(data):

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_csize, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #
    # hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    # W1*x1 + b1 = y1
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    # W2*y1 + b2 = y2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    # W3*y2 + b3 = y3
    # l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)

    # OW*y3 + ob = o
    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # difference between prediction and true solution
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # cost, _ = tf.metrics.recall_at_k(y, prediction, k=20)
    # https://stackoverflow.com/a/44801217
    # cost, _ = tf.metrics.mean(tf.nn.in_top_k(predictions=prediction, targets=y, k=20))
    # cost = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=prediction, targets=y, k=20), tf.float32))
    #                       learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost, y)
    # = cycles of feed forward + backpropagation
    hm_epochs = 10

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        # <class 'list'>
        # <class 'list'>
        print(type(train.x))
        print(type(train.y))
        # Testing
        for epoch in range(hm_epochs):
            epoch_loss = 0
            print(train.num_examples)
            for _ in range(epoch_size):  # _ shorthand for  variable we don't care about
                if _ % (epoch_size//10) == 0:
                    print(str(100*(_/epoch_size))+' % complete')
                epoch_x, epoch_y = train.next_batch(batch_size)
                _, c = sesh.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            #  print(str(c)+' loss ')
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        test_x, test_y = train.next_batch(batch_size*100)
        recall, reop = tf.metrics.recall(y, prediction)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # TypeError: parameter 'targets' has DataType float32 not in list of allowed values: int32, int64
        # correct_topk = tf.nn.in_top_k(predictions=prediction, targets=y, k=25)

        sesh.run(tf.local_variables_initializer())  # init recall

        print('Recall:', reop.eval({x: test_x, y: test_y}))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        # print('CorrectTopk:', correct_topk.eval({x: test_x, y: test_y}))
        # (10000, 784)
        # (10000, 10)
        # print(mnist.test.images.shape)
        # print(mnist.test.labels.shape)
# results FFN
# n_nodes_hl1 = 500
# n_nodes_hl2 = 500
# n_nodes_hl3 = 500
# hm_epochs 10
# n_classes = 52739    # number of outputs
# batch_size = 10  # batch_size images at one time is fed to the network
# n_csize = 52739
# epoch_size = 5000
# default loss function
# test_size 1000
# lowest loss in test epoch around 41000
# Accuracy: 0.017

# same accuracy with FNN on
# n_nodes_hl1 = 2000
# n_nodes_hl2 = 2000
# no hl 3
# same values as above
# lowest loss in test epoch around 41000


train_neural_network(x)
