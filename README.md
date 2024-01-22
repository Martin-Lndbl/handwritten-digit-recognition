# Handwritten Digit Recognition
Just another repository using CNN to classify handwritten digits

## Overview
This repository contains three increasingly improved CNN models for handwritten digit recogition.
* The data used to train `model_01` and `model_02` is not categoricalized
* `model_03` is trained on categoricalized data  

## Usage
* `model_01.py` to `model_03.py` compile and train the respective model, creating a .kreas file
* `test_sparse.py model_XX.keras` can be used to test the accuracy and loss for models trained on not categoricalized data
* `test_categorical.py model_XX.keras` can be used to test the accuracy and loss for models trained on categoricalized data
* `predicy.py model_XX.keras` predicts the number written on images in the `./img` folder while visualizing the png
