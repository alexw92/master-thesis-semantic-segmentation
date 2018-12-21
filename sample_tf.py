import tensorflow as tf
import tensorflow.keras as keras
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
tbCallBack = keras.callbacks.TensorBoard(log_dir='./sample_tf_log', histogram_freq=1, write_graph=True, write_images=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1, epochs=5, callbacks=[tbCallBack])
model.evaluate(x_test, y_test)