import cv2 as cv
import mediapipe as mp
import numpy as np
from constants import *
from eye_tracker import EyeTracker

class FaceMeshDetector:
    def __init__(self, min_detection_confidence, min_tracking_confidence):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mesh_points = None
        self.mesh_points_3D = None

    def detect_face_mesh(self, image):
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            self.mesh_points = np.array(
                [[p.x, p.y] for p in results.multi_face_landmarks[0].landmark]
            )
            self.mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
    
    def process_image(self, frame):
        img_h, img_w = frame.shape[:2]
        # getting the head pose estimation 3d points
        head_pose_points_3D = np.multiply(
            self.mesh_points_3D[indices_pose], [img_w, img_h, 1]
        )
        head_pose_points_2D = self.mesh_points[indices_pose]

        # collect nose three dimension and two dimension points
        nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
        nose_2D_point = head_pose_points_2D[0]

        # create the camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array(
            [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
        )

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
        head_pose_points_3D = head_pose_points_3D.astype(np.float64)
        head_pose_points_2D = head_pose_points_2D.astype(np.float64)
        # Solve PnP
        success, rot_vec, trans_vec = cv.solvePnP(
            head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
        )
        # Get rotational matrix
        rotation_matrix, jac = cv.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

        # Get the y rotation degree
        angle_x = angles[0] * 360
        angle_y = angles[1] * 360
        z = angles[2] * 360

        # if angle cross the values then
        threshold_angle = 10
        # See where the user's head tilting
        if angle_y < -threshold_angle:
            face_looks = "Left"
        elif angle_y > threshold_angle:
            face_looks = "Right"
        elif angle_x < -threshold_angle:
            face_looks = "Down"
        elif angle_x > threshold_angle:
            face_looks = "Up"
        else:
            face_looks = "Forward"
        if SHOW_ON_SCREEN_DATA:
            cv.putText(
                frame,
                f"Face Looking at {face_looks}",
                (img_w - 400, 80),
                cv.FONT_HERSHEY_TRIPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
        # Display the nose direction
        nose_3d_projection, jacobian = cv.projectPoints(
            nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
        )

        p1 = nose_2D_point
        p2 = (
            int(nose_2D_point[0] + angle_y * 10),
            int(nose_2D_point[1] - angle_x * 10),
        )

        cv.line(frame, p1, p2, (255, 0, 255), 3)

        
        # Display all facial landmarks if enabled
        if SHOW_ALL_FEATURES:
            for point in self.mesh_points:
                cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
        # Process and display eye features
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(self.mesh_points[LEFT_EYE_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(self.mesh_points[RIGHT_EYE_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        # Highlighting the irises and corners of the eyes
        cv.circle(
            frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
        )  # Left iris
        cv.circle(
            frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
        )  # Right iris
        cv.circle(
            frame, self.mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
        )  # Left eye right corner
        cv.circle(
            frame, self.mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
        )  # Left eye left corner
        cv.circle(
            frame, self.mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
        )  # Right eye right corner
        cv.circle(
            frame, self.mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
        )  # Right eye left corner
        # Function to calculate vector position
        def vector_position(point1, point2):
            x1, y1 = point1.ravel()
            x2, y2 = point2.ravel()
            return x2 - x1, y2 - y1
        # Calculating relative positions
        l_dx, l_dy = vector_position(self.mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
        r_dx, r_dy = vector_position(self.mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)
        return l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy