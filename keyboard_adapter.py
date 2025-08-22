import time
import pyautogui

class KeyboardAdapter:
    
    def __init__(self, app_config):
        self.app_config = app_config
        self.key_time = 0.0
        pyautogui.PAUSE = 0

    # Releases the specified keys.
    #
    # Args:
    #     keys: A list of keys to release.
    def _key_up(self, keys):
        for key in keys:
            try:
                pyautogui.keyUp(key)
            except Exception as e:
                print(f"Error releasing key '{key}': {e}")


    # Presses and holds the specified keys.
    #
    # Args:
    #     keys: A list of keys to press and hold.
    def _key_down(self, keys):
        for key in keys:
            try:
                pyautogui.keyDown(key)
            except Exception as e:
                print(f"Error pressing key '{key}': {e}")


    # Presses and releases the specified keys.
    #
    # Args:
    #     keys: A list of keys to press and release.
    def _press_keys(self, keys):
        for key in keys:
            try:
                pyautogui.press(key)
            except Exception as e:
                print(f"Error pressing key '{key}': {e}")


    # Handles the key press/release logic based on the previous and new gesture.
    #
    # Args:
    #     prev_gesture_id: The ID of the previous gesture.
    #     new_gesture_id: The ID of the new gesture.
    def handle_gesture_key(self, prev_gesture_id, new_gesture_id):
        if prev_gesture_id is not None:
            prev_keys = self.app_config.get_keys_for_gesture(prev_gesture_id)
            if prev_keys:
                self._key_up(prev_keys)
 

        if new_gesture_id is not None:
            new_keys = self.app_config.get_keys_for_gesture(new_gesture_id)
            behavior = self.app_config.get_behavior_for_gesture(new_gesture_id)
            
            if behavior == 'hold' and new_keys:
                self._key_down(new_keys)
            elif behavior == 'press' and new_keys:
                self._press_keys(new_keys)


    # Releases all keys that are currently pressed.
    def release_all_keys(self):
        try:
            all_keys = self.app_config.get_all_keys()
            self._key_up(all_keys)
        except Exception as e:
            print(f"Error releasing keys: {e}")
