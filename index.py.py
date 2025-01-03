import cv2
import numpy as np
import tensorflow as tf

# Load the trained TensorFlow model
model = tf.keras.models.load_model('D:\ML workspace\rotten\model.h5')

# Define the labels
labels = {0:'freshapples', 1:'freshbanana', 2:'freshoranges', 3:'rottenapples', 4:'rottenbanana', 5:'rottenoranges'}

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame: resize, normalize, and expand dimensions
    input_frame = cv2.resize(frame, (150, 150))  # Resize to match model input size
    input_frame = tf.keras.applications.mobilenet_v2.preprocess_input(input_frame)  # Normalize using MobileNetV2 preprocessing
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Perform inference
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = labels[predicted_class]

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
