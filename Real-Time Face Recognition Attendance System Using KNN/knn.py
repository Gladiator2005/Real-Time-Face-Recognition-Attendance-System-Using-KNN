from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch




# Disable OpenCL to prevent potential issues with certain systems
cv2.ocl.setUseOpenCL(False)




# Function to make the system speak (Windows specific)
def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)




# Load the face data and labels from the pickled files
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

print('Shape of Faces matrix --> ', FACES.shape)





# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)





# Check if the Attendance directory exists, create it if not
attendance_dir = "Attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

COL_NAMES = ['NAME', 'TIME']





# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')




# Start capturing video for face recognition
while True:
    ret, frame = video.read()  # Capture frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]  # Crop the detected face
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Resize and flatten the image
        output = knn.predict(resized_img)  # Predict the person's identity using the KNN model
        
        # Get the current timestamp and date
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        # Draw rectangles and text around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        # Prepare attendance record
        attendance = [str(output[0]), str(timestamp)]
    
    # Display the video frame with rectangles and text
    cv2.imshow("Frame", frame)
    
    # Save attendance when 'o' is pressed
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        attendance_file = os.path.join(attendance_dir, f"Attendance_{date}.csv")
        file_exists = os.path.isfile(attendance_file)
        
        with open(attendance_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)
    
    # Exit loop when 'q' is pressed
    if k == ord('q'):
        break




# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
