import sys
import tensorflow as tf


model = tf.keras.models.load_model(sys.argv[1])
model.summary()
