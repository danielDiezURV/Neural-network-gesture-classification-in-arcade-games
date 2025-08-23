import numpy as np
import mediapipe as mp

class DataPreprocessor:

    def __init__(self):
        self.mp_hands = mp.solutions.hands


    # Processes a single hand vector to normalize and flatten it.
    #
    # Args:
    #     hand_vector (list of tuples): A list of (x, y, z) tuples for each of the 21 hand landmarks.
    #
    # Returns:
    #     np.ndarray: A flattened and normalized 1D numpy array of 60 elements (20 landmarks × 3 coordinates), 
    #                 or None if the input vector is invalid.
    def process(self, hand_vector):
        if not hand_vector or len(hand_vector) != 21:
            return None

        landmarks = np.array(hand_vector)
        normalized_landmarks = self._normalize(landmarks)
        return self._flatten(normalized_landmarks)


    # Normalizes hand landmarks to be translation and scale-invariant.
    #
    # The normalization process consists of three steps:
    # 1. Position Normalization: Centers the landmarks around the wrist (landmark 0).
    # 2. Scale Normalization: Scales the landmarks based on the maximum distance
    #     from the wrist to any other landmark, making the gesture scale-invariant.
    # 3. Wrist Removal: Removes the wrist landmark since it becomes (0, 0, 0) after normalization.
    #
    # Args:
    #     landmarks (np.ndarray): A (21, 3) numpy array of hand landmarks.
    #
    # Returns:
    #     np.ndarray: A (20, 3) numpy array of normalized hand landmarks without the wrist (0.0, 0.0, 0.0).
    def _normalize(self, landmarks):
        wrist_lm = landmarks[self.mp_hands.HandLandmark.WRIST]
        normalized_landmarks = landmarks - wrist_lm

        max_dist = np.max(np.linalg.norm(normalized_landmarks, axis=1))
        if max_dist > 0:
            normalized_landmarks /= max_dist
        
        # Remove the wrist landmark (index 0) since it's always (0, 0, 0) after normalization
        return normalized_landmarks[1:]


    # Flattens the normalized landmark matrix and rounds the values.
    #
    # The process involves two steps:
    # 1. Flattening: Converts the (20, 3) matrix into a 1D vector of 60 elements.
    # 2. Rounding: Truncates the floating-point values to 4 decimal places for consistency.
    #
    # Args:
    #     normalized_landmarks (np.ndarray): A (20, 3) numpy array of normalized hand landmarks.
    #
    # Returns:
    #     np.ndarray: A flattened and rounded 1D numpy array of 60 elements.
    def _flatten(self, normalized_landmarks):
        flattened_vector = normalized_landmarks.flatten()
        rounded_vector = np.round(flattened_vector, 4)
        return rounded_vector

