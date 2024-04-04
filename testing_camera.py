from tensorflow.keras.models import load_model
import time

model = load_model("My_Model_2.h5")

vid = cv2.VideoCapture(0)

#label your classes here. Make sure that the order of labels follows the order that the folders show up in your training set
labels =  [ 'Class_1','Class_2','Class_3',...]
image_count = 0
if not vid.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = vid.read()
    #resizes the image to match the size that your model is looking for
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(frame, (256, 256)) 
    resized_image = resized_image / 255.0 
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    f_val = 5  # Adjust this value as needed
    vid.set(cv2.CAP_PROP_FOCUS, f_val)

    #makes prediction here
    prediction = model.predict(np.expand_dims(resized_image, axis=0))
    
    predicted_class_index = np.argmax(prediction)
    predicted_label = labels[predicted_class_index]
  
    #prints out prediction here on the camera screen
    cv2.putText(frame, "Prediction: {}".format(predicted_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    cv2.imshow('Input', frame)
  """
  if you want it to only capture a frame every few seconds, you can have it wait here between iterations
    #time.sleep(5)
  """
  
    

    c = cv2.waitKey(1)
    if c == 27:
        break

vid.release()
cv2.destroyAllWindows()
