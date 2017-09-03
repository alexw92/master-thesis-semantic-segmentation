# imports
import tensorflow as tf;

# start

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

# print(node1, node2)
sess = tf.Session()

# print(sess.run([node1,node2]))

# free resources
# sess.close()

# python with statement
# Useful link with explanation: http://effbot.org/zone/python-with-statement.htm
# TODO include this in markdown doc file
with tf.Session() as sess:
    output = sess.run([node1, node2])
    print(output)
