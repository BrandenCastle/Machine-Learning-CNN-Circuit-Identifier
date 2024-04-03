from tensorflow.keras.models import load_model

#load_model
model = load_model("My_Model.h5")


import tensorflow as tf
import numpy as np
import cv2

#create labels and make sure to match the labels' location with the way that they appear in training
labels =  [ 'Class_1','Class_2','Class_3'...]

# Load your example image
image = cv2.imread("image_1.jpg") 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

# Preprocess the image
image = cv2.resize(image, (256,256))  # Resize the image to the input shape of your model
image = image / 255.0  # Normalize pixel values to [0, 1]
image = np.expand_dims(image, axis=0) 

# Make predictions
predictions = model.predict(image)

#predict class and assign each class to its respective label
predicted_class_index = np.argmax(predictions)
predicted_label = labels[predicted_class_index]
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_label)
print("Probabilities:", predictions)
