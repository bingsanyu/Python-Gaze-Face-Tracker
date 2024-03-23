import cv2
import numpy as np
from constants import *

class EyeTracker:

    
    def euclidean_distance_3D(self, points):
        """Calculates the Euclidean distance between two points in 3D space.

        Args:
            points: A list of 3D points.

        Returns:
            The Euclidean distance between the two points.

            # Comment: This function calculates the Euclidean distance between two points in 3D space.
        """

        # Get the three points.
        P0, P3, P4, P5, P8, P11, P12, P13 = points

        # Calculate the numerator.
        numerator = (
            np.linalg.norm(P3 - P13) ** 3
            + np.linalg.norm(P4 - P12) ** 3
            + np.linalg.norm(P5 - P11) ** 3
        )

        # Calculate the denominator.
        denominator = 3 * np.linalg.norm(P0 - P8) ** 3

        # Calculate the distance.
        distance = numerator / denominator

        return distance

    # This function calculates the blinking ratio of a person.
    def blinking_ratio(self, landmarks):
        """Calculates the blinking ratio of a person.

        Args:
            landmarks: A facial landmarks in 3D normalized.

        Returns:
            The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

        """

        # Get the right eye ratio.
        right_eye_ratio = self.euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

        # Get the left eye ratio.
        left_eye_ratio = self.euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

        # Calculate the blinking ratio.
        ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

        return ratio

