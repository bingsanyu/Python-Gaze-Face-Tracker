"""
Face Landmark and Eye Tracker System
GitHub: https://github.com/alireza787b/Python-Gaze-Face-Tracker
Email: p30planets@gmail.com
LinkedIn: https://www.linkedin.com/in/alireza787b
Date: November 2023

Description:
This Python-based application is designed for advanced eye tracking, utilizing OpenCV and MediaPipe. 
It provides real-time iris tracking, logging of eye position data, and features socket communication for data transmission. 
The application is highly customizable, allowing users to control various parameters and logging options.

Features:
- Real-Time Eye Tracking: Utilizes webcam input for tracking and visualizing iris and eye corner positions.
- Data Logging: Records tracking data in CSV format, including timestamps and positional data, with an option to log all 468 facial landmarks.
- Socket Communication: Transmits eye tracking data via UDP sockets, configurable through user-defined IP and port.
- Customizable Settings: Includes multiple parameters for customizing tracking and logging behavior.

Requirements:
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Dependencies: math, socket, argparse, time, csv, datetime, os

Inspiration:
Initially inspired by Asadullah Dal's iris segmentation project: https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python


Parameters:
- SERVER_IP & SERVER_PORT: Configurable IP address and port for UDP socket communication. 
  These parameters define where the eye tracking data is sent via the network.
- DEFAULT_WEBCAM: Specifies the default webcam source number. If a webcam number is provided as a command-line argument, 
  that number is used; otherwise, the default value here is used.
- PRINT_DATA: When set to True, the program prints eye tracking data to the console. Set to False to disable printing.
- SHOW_ALL_FEATURES: If True, the program shows all facial landmarks. Set to False to display only the eye positions.
- LOG_DATA: Enables logging of eye tracking data to a CSV file when set to True.
- LOG_ALL_FEATURES: When True, all 468 facial landmarks are logged in the CSV file.


Usage:
Run the script in a Python environment with the necessary dependencies installed. The script accepts command-line arguments for camera source configuration. The application displays a real-time video feed with eye tracking visualization. Press 'q' to quit the application and save the log data.

Note:
This project is intended for educational and research purposes in fields like aviation, human-computer interaction, and more.
"""



import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os

# User-configurable parameters
PRINT_DATA = True  # Enable/disable data printing
DEFAULT_WEBCAM = 1  # Default webcam number
SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
LOG_DATA = True  # Enable logging to CSV
LOG_ALL_FEATURES = False  # Log all facial landmarks if True
LOG_FOLDER = "logs"  # Folder to store log files

# Server configuration
SERVER_IP = "127.0.0.1"  # Set the server IP address (localhost)
SERVER_PORT = 7070       # Set the server port


# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument("-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM))
args = parser.parse_args()

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]    # Left eye Left Corner
L_H_RIGHT = [133]  # Left eye Right Corner
R_H_LEFT = [362]   # Right eye Left Corner
R_H_RIGHT = [263]  # Right eye Right Corner

# Server address for UDP socket communication
SERVER_ADDRESS = (SERVER_IP, 7070)

# Function to calculate vector position
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1

# Initializing MediaPipe face mesh and camera
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)

# Initializing socket for data transmission
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Preparing for CSV logging
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Column names for CSV file
column_names = ['Timestamp (ms)', 'Left Eye Center X', 'Left Eye Center Y', 'Right Eye Center X', 'Right Eye Center Y', 
                'Left Iris Relative Pos Dx', 'Left Iris Relative Pos Dy', 'Right Iris Relative Pos Dx', 'Right Iris Relative Pos Dy']
if LOG_ALL_FEATURES:
    column_names.extend([f'Landmark_{i}_X' for i in range(468)] + [f'Landmark_{i}_Y' for i in range(468)])

# Main loop for video capture and processing
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flipping the frame for a mirror effect
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # Display all facial landmarks if enabled
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            # Process and display eye features
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)  # Left iris
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)  # Right iris
            cv.circle(frame, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)  # Left eye right corner
            cv.circle(frame, mesh_points[L_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)    # Left eye left corner
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)  # Right eye right corner
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)     # Right eye left corner

            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[L_H_LEFT], center_left)
            r_dx, r_dy = vector_position(mesh_points[R_H_LEFT], center_right)

            # Printing data if enabled
            if PRINT_DATA:
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")

            # Logging data
            if LOG_DATA:
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy]
                if LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                csv_data.append(log_entry)

            # Sending data through socket
            packet = np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32)
            iris_socket.sendto(bytes(packet), SERVER_ADDRESS)

        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            if PRINT_DATA:
                print("Exiting program...")
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()
    iris_socket.close()
    if PRINT_DATA:
        print("Program exited successfully.")

    # Writing data to CSV file
    if LOG_DATA:
        if PRINT_DATA:
            print("Writing data to CSV...")
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv")
        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)  # Writing column names
            writer.writerows(csv_data)     # Writing data rows
        if PRINT_DATA:
            print(f"Data written to {csv_file_name}")