## Eye Disease Image Classifier 
This project uses a TensorFlow deep learning model to classify OCT (Optical Coherence Tomography) eye images into four disease categories:

CNV
DME
DRUSEN
NORMAL

The model is trained using OCT images and can be used to predict eye conditions from new images.

##  Dataset Structure
Your dataset should follow this folder format:
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
Each folder contains images belonging to that class.


##  Installation
Install required libraries:
pip install -r requirements.txt

requirements.txt
tensorflow
matplotlib
pandas
seaborn
numpy
streamlit

##  Training the Model

Use the notebook:
training_model.ipynb to train your deep learning model. Training may take long if you do not have a GPU.

##  Making Predictions

You can test your trained model using:
app.py (Streamlit web app)
Model_Prediction.ipynb
Upload an OCT image and the model will classify it into one of the four categories.

## ⚠️ Note About Model Accuracy

This project is mainly for learning and demonstration.  
Because the model is trained on a limited dataset and training on CPU takes many hours (up to 24 hours), the model may sometimes:

- Predict the same class repeatedly  
- Give different accuracy each time it is trained  
- Perform differently on new images  

These issues are normal when training deep learning models without a GPU and with smaller datasets.


