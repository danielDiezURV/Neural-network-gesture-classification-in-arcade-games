import cv2
import mediapipe as mp

class GestureDetector:

    # Initializes the GestureDetector.
    #
    # Args:
    #     max_hands (int): Maximum number of hands to detect.
    #     min_detection_confidence (float): Minimum confidence value for hand detection.
    def __init__(self, max_hands=2, min_detection_confidence=0.7):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None


    # Processes a single video frame to find hands.
    #
    # Args:
    #     image: The video frame (in BGR format).
    #
    # Returns:
    #     The processed image with hand landmarks drawn.
    def process_frame(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return image


    # Extracts the landmark vectors for detected left and right hands.
    #
    # Returns:
    #     A tuple containing (left_hand_vector, right_hand_vector).
    #     Vectors will be None if the corresponding hand is not detected.
    def get_hand_vectors(self):
        left_hand_vector = None
        right_hand_vector = None

        if self.results and self.results.multi_hand_landmarks and self.results.multi_handedness:
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                handedness = self.results.multi_handedness[idx].classification[0].label
                
                hand_vector = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                if handedness == "Left":
                    left_hand_vector = hand_vector
                else:
                    right_hand_vector = hand_vector

        return left_hand_vector, right_hand_vector


    # Draws the gesture names on the image.
    #
    # Args:
    #     image: The image to draw on.
    #     left_gesture_name (str): The name of the gesture for the left hand.
    #     right_gesture_name (str): The name of the gesture for the right hand.
    #
    # Returns:
    #     The image with the gesture names drawn.
    def draw_gesture_names(self, image, left_gesture_name, right_gesture_name):
        if self.results and self.results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                handedness = self.results.multi_handedness[idx].classification[0].label
                gesture_name = left_gesture_name if handedness == "Left" else right_gesture_name
                
                if gesture_name:
                    # Get the coordinates of the wrist to position the text
                    x = int(hand_landmarks.landmark[0].x * image.shape[1])
                    y = int(hand_landmarks.landmark[0].y * image.shape[0])
                    cv2.putText(image, gesture_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image

