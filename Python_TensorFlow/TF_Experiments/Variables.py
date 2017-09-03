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
# x = in; y = out
x = tf.placeholder(tf.float32)

linear_model = W * x + b

y = tf.placeholder(tf.float32)

# Loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Define vars
init = tf.global_variables_initializer()

# Create session object
sesh = tf.Session()
sesh.run(init)

# train the model
for i in range(1000):
    sesh.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sesh.run([W, b]))
# print(sesh.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# print(b);

# cleanup
sesh.close()
