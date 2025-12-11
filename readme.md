Eye Disease Image Classifier

This project uses a deep learning model (TensorFlow) to classify eye OCT images into four groups:

CNV

DME

DRUSEN

NORMAL

The model learns from images in the training folder and is tested using the validation and test folders.

Dataset Structure

Your dataset must be organized like this:

train/
    CNV/
    DME/
    DRUSEN/
    NORMAL/
val/
    CNV/
    DME/
    DRUSEN/
    NORMAL/
test/
    CNV/
    DME/
    DRUSEN/
    NORMAL/


Each folder contains images for that class.

Installation

Install needed libraries:

pip install -r requirements.txt


requirements.txt

tensorflow==2.15.0
matplotlib
pandas
seaborn

Training the Model

Use the notebook training_model.ipynb to train the model.
Training may take a long time if you do not have a GPU.

Making Predictions

Use app.py or Model_Prediction.ipynb to test the model on new images.

Common Issues

The model sometimes predicts the same class a lot.

Accuracy may change each time you train.

Training can take many hours on a slow computer.

These issues are normal for beginner deep learning projects.

Improving Results

Simple ways to improve the model:

Add more images

Use image augmentation

Use transfer learning (pretrained models)

Reduce image size to speed up training