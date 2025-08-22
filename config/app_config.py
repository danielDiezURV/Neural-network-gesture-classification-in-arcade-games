import json
import os

APP_CONFIG_PATH = 'config/app_config.json'

class AppConfig:
    
    def __init__(self, key_bindings_path='data/key_bindings.json'):
        self.app_config_data = self._load_app_config()
        self.key_bindings_data = self._load_key_bindings(key_bindings_path)

    def _load_app_config(self):
        if os.path.exists(APP_CONFIG_PATH):
            try:
                with open(APP_CONFIG_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading app configuration: {e}")
                return {}
        else:
            print(f"Warning: App config file not found at {APP_CONFIG_PATH}")
            return {}
    
    def _load_key_bindings(self, key_bindings_path):
        if os.path.exists(key_bindings_path):
            try:
                with open(key_bindings_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading key bindings: {e}")
                return {}
        else:
            print(f"Warning: Key bindings file not found at {key_bindings_path}")
            return {}
    
    def get_app_config(self, key=None):
        if key is None:
            return self.app_config_data
        return self.app_config_data.get(key, {})
    
    def get_model_path(self):
        return self.app_config_data.get('NEURAL_NETWORK_CONFIG').get('MODEL_PATH')

    def get_key_bindings(self, gesture_id=None):
        if gesture_id is None:
            return self.key_bindings_data
        return self.key_bindings_data.get(str(gesture_id), {})
    
    def get_gesture_name(self, gesture_id):
        if gesture_id is None or gesture_id == -1:
            return "unknown"
        key_binding = self.get_key_bindings(gesture_id)
        return key_binding.get("gesture", "unknown")
    
    def get_keys_for_gesture(self, gesture_id):
        if gesture_id is None or gesture_id == -1:
            return []
        key_binding = self.get_key_bindings(gesture_id)
        return key_binding.get("keys", [])
    
    def get_behavior_for_gesture(self, gesture_id):
        if gesture_id is None or gesture_id == -1:
            return None
        key_binding = self.get_key_bindings(gesture_id)
        return key_binding.get("behavior")
    
    def get_all_keys(self):
        all_keys = set()
        for gesture_data in self.key_bindings_data.values():
            keys = gesture_data.get("keys", [])
            all_keys.update(keys)
        return list(all_keys)
    
    def get_all_gesture_ids(self):
        return [int(gesture_id) for gesture_id in self.key_bindings_data.keys()]
    
    def get_hand_gesture_config(self):
        return self.get_app_config('HAND_GESTURE_CONFIG')
    
    def get_neural_network_config(self):
        return self.get_app_config('NEURAL_NETWORK_CONFIG')
    