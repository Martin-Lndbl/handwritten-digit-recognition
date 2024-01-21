import sys
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

model = tf.keras.models.load_model(sys.argv[1])
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
