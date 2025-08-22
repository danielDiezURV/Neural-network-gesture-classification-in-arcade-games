import cv2
import time
import json
from gesture_detector import GestureDetector
from data_preprocessor import DataPreprocessor
from gesture_classifier import GestureClassifier
from keyboard_adapter import KeyboardAdapter
from config.app_config import AppConfig
import os

def main():
    # Configuration
    KEY_BINDINGS_CONFIG = "data/key_bindings_default.json"
    
    # Load configuration
    app_config = AppConfig(KEY_BINDINGS_CONFIG)
    gesture_config = app_config.get_hand_gesture_config()
    max_hands = gesture_config.get('MAX_HANDS')
    min_detection_confidence = gesture_config.get('MIN_DETECTION_CONFIDENCE')
    detection_rate = gesture_config.get('DETECTION_RATE')
    model_path = app_config.get_model_path()

    detector = GestureDetector(max_hands, min_detection_confidence)
    keyboard_adapter = KeyboardAdapter(app_config)
    data_preprocessor = DataPreprocessor()
    classifier = GestureClassifier()
    classifier.load_model(model_path)

    # Camera setup with optimized settings for better performance
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
    cap.set(cv2.CAP_PROP_FPS, 30)  
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

    last_print_time = time.time()

    # Track previous gesture predictions
    prev_left_gesture = None
    prev_right_gesture = None

    try:
        while cap.isOpened():
            success, image = cap.read()
            current_time = time.time()
            if current_time - last_print_time >= detection_rate:

                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.flip(image, 1)
                processed_image = detector.process_frame(image)
                left_hand, right_hand = detector.get_hand_vectors()
                norm_left_hand = data_preprocessor.process(left_hand)
                norm_right_hand = data_preprocessor.process(right_hand)

                gesture_id_left = classifier.predict(norm_left_hand)
                gesture_id_right = classifier.predict(norm_right_hand)

                right_hand_changed = (gesture_id_right != prev_right_gesture)
                left_hand_changed = (gesture_id_left != prev_left_gesture)

                if right_hand_changed or left_hand_changed:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # Handle right hand gesture changes
                    if right_hand_changed and gesture_id_right is not None:
                        keyboard_adapter.handle_gesture_key(prev_right_gesture, gesture_id_right)
                        print(f"Right hand: {gesture_id_right}, {app_config.get_gesture_name(gesture_id_right)}")
                        prev_right_gesture = gesture_id_right

                    # Handle left hand gesture changes
                    if left_hand_changed and gesture_id_left is not None:
                        keyboard_adapter.handle_gesture_key(prev_left_gesture, gesture_id_left)
                        print(f"Left hand: {gesture_id_left}, {app_config.get_gesture_name(gesture_id_left)}")
                        prev_left_gesture = gesture_id_left

                # Get gesture names
                left_gesture_name = app_config.get_gesture_name(gesture_id_left) if gesture_id_left is not None else ""
                right_gesture_name = app_config.get_gesture_name(gesture_id_right) if gesture_id_right is not None else ""

                # Draw gesture names on the image
                processed_image = detector.draw_gesture_names(processed_image, left_gesture_name, right_gesture_name)

                last_print_time = current_time
                cv2.imshow('Hand Tracking', processed_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release any remaining pressed keys when exiting
        print("Releasing remaining keys...")
        keyboard_adapter.release_all_keys()

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cap.release()

if __name__ == "__main__":
    main()
