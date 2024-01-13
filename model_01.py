import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()

# Convert 2D Matrix into Vector
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# relu = rectify linear unit (range: [0, inf))
model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))

# Map to actual output
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('model_01.keras')
