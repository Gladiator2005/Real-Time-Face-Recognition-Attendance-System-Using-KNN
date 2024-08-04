# Face Adding Code
import cv2
import pickle
import numpy as np
import os

# Disable OpenCL
cv2.ocl.setUseOpenCL(False)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")
num_images = 100  # Number of images to collect per person

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= num_images and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == num_images:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(num_images, -1)

# Save face data and names
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if 'names.pkl' not in os.listdir(data_dir):
    names = [name] * num_images
    with open(data_dir + 'names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open(data_dir + 'names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * num_images
    with open(data_dir + 'names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir(data_dir):
    with open(data_dir + 'faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(data_dir + 'faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(data_dir + 'faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)


# KNN and Attendance Marking Code
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(text)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load label and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)


