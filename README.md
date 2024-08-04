Real-Time Face Recognition Attendance System Using KNN

A real-time face recognition attendance system using K-Nearest Neighbors (KNN) and OpenCV.

Table of Contents:
  Introduction
  Features
  Installation
  Usage
  Project Structure
  Contributing
  License


Introduction:
  This project implements a real-time face recognition attendance system using the K-Nearest Neighbors (KNN) algorithm. The system captures faces using a webcam, processes them using OpenCV, and stores the data for future recognition. The attendance is marked automatically when a recognized face appears in front of the camera, and the data is stored in CSV files.



Features:
  Real-time face detection and recognition using KNN.
  Attendance marking with timestamps.
  Data storage in CSV files.
  Easy to extend and customize.
  Voice feedback for successful attendance marking (Windows only).


Installation:
  Prerequisites
  Python 3.6 or later
  OpenCV
  NumPy
  scikit-learn
  pywin32 (for Windows text-to-speech)


Clone the Repository:
git clone https://github.com/Gladiator2005/Real-Time-Face-Recognition-Attendance-System-Using-KNN.git
cd Real-Time-Face-Recognition-Attendance-System-Using-KNN


Install Dependencies:
You can install the required Python packages using pip

pip install -r requirements.txt




Set Up
Ensure you have a webcam connected to your system.
Place the Haar Cascade XML file ('haarcascade_frontalface_default.xml') in the 'data/' directory.

Usage
Face Data Collection
To add new faces to the system, run the face_add.py script:
python face_add.py
Follow the prompts to enter your name and capture images.

Face Recognition and Attendance
To start the face recognition and attendance system, run the knn.py script:
python knn.py
Press 'o' to mark attendance and save it to a CSV file.
Press 'q' to quit the program.

Project Structure:
Real-Time-Face-Recognition-Attendance-System-Using-KNN/
│
├── data/                          # Directory to store face data and Haar Cascade XML
│   ├── faces_data.pkl             # Pickled file for face data
│   ├── names.pkl                  # Pickled file for names associated with face data
│   └── haarcascade_frontalface_default.xml # Haar Cascade classifier for face detection
│
├── Attendance/                    # Directory to store attendance CSV files
│   └── Attendance_<date>.csv      # Generated attendance file
│
├── face_add.py                    # Script for adding face data
├── knn.py                         # Script for face recognition and attendance
├── requirements.txt               # List of required Python packages
└── README.md                      # Project README file

Contributing:
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
