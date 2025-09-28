# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
# predictions = model(x_train[:1]).numpy()
# tf.nn.softmax(predictions).numpy()
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test,  y_test, verbose=2)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)
print(c)
