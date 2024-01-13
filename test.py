import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

model = tf.keras.models.load_model('NN_model.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
