import numpy as np
import cv2 as cv
from constants import *

def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch

class HeadPoseEstimator:
    
    def estimate_head_pose(self, landmarks, image_size):
        # Scale factor based on user's face width (assumes model face width is 150mm)
        scale_factor = USER_FACE_WIDTH / 150.0
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
            (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
            (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
            (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
            (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
        ])
        

        # Camera internals
        focal_length = image_size[1]
        center = (image_size[1]/2, image_size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1))

        # 2D image points from landmarks, using defined indices
        image_points = np.array([
            landmarks[NOSE_TIP_INDEX],            # Nose tip
            landmarks[CHIN_INDEX],                # Chin
            landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
            landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
            landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
            landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
        ], dtype="double")


            # Solve for pose
        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)

        # Combine rotation matrix and translation vector to form a 3x4 projection matrix
        projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

        # Decompose the projection matrix to extract Euler angles
        _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()[:3]
        # Normalize the pitch angle
        pitch = normalize_pitch(pitch)
        return pitch, yaw, roll