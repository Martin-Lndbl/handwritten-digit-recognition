import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if sys.argv.__len__() != 2:
    print("Provide a model")
    exit()

model = tf.keras.models.load_model(sys.argv[1])

i = 0

while os.path.isfile(f"img/{i}.png"):
    img = cv2.imread(f"img/{i}.png")[:,:,0]
    img = np.invert(np.array([img]))
    pred = model.predict(img)
    print(f"Prediction: {np.argmax(pred)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    i += 1
