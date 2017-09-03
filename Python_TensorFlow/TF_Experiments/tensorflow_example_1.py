import tensorflow as tf

# Build a graph
a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

# Launch the graph in a session

sess = tf.Session()

File_Writer = tf.summary.FileWriter('C:\\Uni\\Masterstudium\\Masterarbeit\\Python_TensorFlow\\TF_Experiments\\graph'
                                    , sess.graph)

# Evaluate the tensor 'c'
print(sess.run(c))

sess.close()

# after running this code, open cmd in folder Python_TensorFlow and enter "tensorboard --logdir="TF_Experiments"
# then open "localhost:6006" in the browser
