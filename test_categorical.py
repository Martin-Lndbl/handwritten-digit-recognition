import sys
import tensorflow as tf
from keras.utils import normalize, to_categorical

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_test = normalize(x_test, axis=1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = to_categorical(y_test, num_classes=10)

model = tf.keras.models.load_model(sys.argv[1])
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
