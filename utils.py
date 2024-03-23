import cv2 as cv
import numpy as np
import argparse
import socket
import os
from constants import *
from datetime import datetime
import csv
import time



def initialize_camera():
    cam_source = int(2)
    cap = cv.VideoCapture(cam_source)
    return cap

def initialize_socket():
    iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return iris_socket

class Logger:
    def __init__(self):
        self.csv_data = []
        if not os.path.exists(LOG_FOLDER):
            os.makedirs(LOG_FOLDER)
        # Column names for CSV file
        self.column_names = [
            "Timestamp (ms)",
            "Left Eye Center X",
            "Left Eye Center Y",
            "Right Eye Center X",
            "Right Eye Center Y",
            "Left Iris Relative Pos Dx",
            "Left Iris Relative Pos Dy",
            "Right Iris Relative Pos Dx",
            "Right Iris Relative Pos Dy",
            "Total Blink Count",
        ]
        # Add head pose columns if head pose estimation is enabled
        if ENABLE_HEAD_POSE:
            self.column_names.extend(["Pitch", "Yaw", "Roll"])
            
        if LOG_ALL_FEATURES:
            self.column_names.extend(
                [f"Landmark_{i}_X" for i in range(468)]
                + [f"Landmark_{i}_Y" for i in range(468)]
            )
        if PRINT_DATA:
            print("Initializing the face mesh and camera...")
            if PRINT_DATA:
                head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
                print(f"Head pose estimation is {head_pose_status}.")
        

    def close_resources(self, cap, iris_socket):
        if PRINT_DATA:
            print("Writing data to CSV...")
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(
            LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv"
        )
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)  # Writing column names
            writer.writerows(self.csv_data)  # Writing data rows
        if PRINT_DATA:
            print(f"Data written to {csv_file_name}")

    # 记录数据
    def log_data(self, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, mesh_points, pitch, yaw, roll):
        timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
        log_entry = [
            timestamp,
            l_cx,
            l_cy,
            r_cx,
            r_cy,
            l_dx,
            l_dy,
            r_dx,
            r_dy,
            TOTAL_BLINKS,
        ]  # Include blink count in CSV
        log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, TOTAL_BLINKS]  # Include blink count in CSV
        
        # Append head pose data if enabled
        if ENABLE_HEAD_POSE:
            log_entry.extend([pitch, yaw, roll])
        self.csv_data.append(log_entry)
        if LOG_ALL_FEATURES:
            log_entry.extend([p for point in mesh_points for p in point])
        self.csv_data.append(log_entry)


    def normalize_pitch(pitch):
        # Normalize pitch angle to be within the range [-90, 90]
        return max(-90, min(90, pitch))