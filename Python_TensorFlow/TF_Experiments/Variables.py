# Tensorflow tut: https://www.youtube.com/watch?v=yX8KuPZCAMo 18:16
# In order to train a model, variables are needed
# W and b should change during training
import tensorflow as tf

# Init variables
# Model parameters
# Setting optimal values for W and b (this process should be done by the neural network)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Inputs and Outputs
# x = in; y = out (expected output)
x = tf.placeholder(tf.float32)

# model output
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# Loss (calculated value - correct value)
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Create var initializer
init = tf.global_variables_initializer()

# Create session object
sesh = tf.Session()
# Define variables with initializer
sesh.run(init)

# tensorboard visualization
# cmd: \TF_Experiments>tensorboard --logdir=. Url:localhost:6006
File_Writer = tf.summary.FileWriter('C:\\Uni\\Masterstudium\\Masterarbeit\\Python_TensorFlow\\TF_Experiments\\graph'
                                    , sesh.graph)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# train the model
for i in range(1000):
    sesh.run(train, {x: x_train, y: y_train})

# evaluate accuracy
curr_W, curr_b, curr_loss = sesh.run([W, b, loss], {x: x_train, y : y_train})

print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss));
# print(sesh.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# print(b);

# cleanup
sesh.close()
