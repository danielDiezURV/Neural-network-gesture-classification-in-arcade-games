import json
import os
from typing import Any, Dict, List, Optional, Set

# Define constants for configuration filenames.
# Paths are relative to this file's location.
_CURRENT_DIR = os.path.dirname(__file__)
_APP_CONFIG_FILENAME = 'app_config.json'
_DEFAULT_KEY_BINDINGS_FILENAME = 'key_bindings_default.json'


class AppConfig:
    """
    Manages loading and accessing application configuration from JSON files.

    This class handles loading the main application settings and the key bindings
    for gestures. It provides a centralized interface for all configuration data,
    ensuring that settings are loaded once and accessed consistently.
    """

    def __init__(self):
        """
        Initializes the AppConfig instance by loading configurations.
        
        It locates and loads the main 'app_config.json' and the default
        key bindings file specified within it.
        """
        app_config_path = os.path.join(_CURRENT_DIR, _APP_CONFIG_FILENAME)
        self.app_config_data = self._load_json_file(app_config_path)

        key_bindings_filename = self.get_hand_gesture_config().get(
            "DEFAULT_KEY_BINDINGS", _DEFAULT_KEY_BINDINGS_FILENAME
        )
        key_bindings_path = os.path.join(_CURRENT_DIR, key_bindings_filename)
        self.key_bindings_data = self._load_json_file(key_bindings_path)

    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Loads a JSON file safely.

        Args:
            file_path (str): The absolute path to the JSON file.

        Returns:
            Dict[str, Any]: The loaded JSON data as a dictionary, or an empty
                            dictionary if the file is not found or is invalid.
        """
        if not os.path.exists(file_path):
            print(f"Warning: Configuration file not found at {file_path}")
            return {}
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading configuration from {file_path}: {e}")
            return {}

    def get_app_config(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves a specific section or the entire application configuration.

        Args:
            key (Optional[str]): The top-level key of the configuration section to retrieve.
                                 If None, returns the entire configuration. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the requested configuration section,
                            or the entire configuration data. Returns an empty dict if the
                            key is not found.
        """
        if key is None:
            return self.app_config_data
        return self.app_config_data.get(key, {})

    def get_model_path(self) -> Optional[str]:
        """
        Retrieves the file path for the trained neural network model.

        Returns:
            Optional[str]: The path to the model file (e.g., 'models/gesture_model.h5'),
                           or None if not defined in the configuration.
        """
        return self.get_neural_network_config().get('MODEL_PATH')

    def get_key_bindings(self, gesture_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieves key bindings for a specific gesture or all key bindings.

        Args:
            gesture_id (Optional[int]): The ID of the gesture to look up. If None,
                                        returns all key bindings. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the key binding data for the specified
                            gesture, or all key bindings. Returns an empty dict if the
                            gesture_id is not found.
        """
        if gesture_id is None:
            return self.key_bindings_data
        return self.key_bindings_data.get(str(gesture_id), {})

    def get_gesture_name(self, gesture_id: int) -> str:
        """
        Retrieves the human-readable name for a given gesture ID.

        Args:
            gesture_id (int): The ID of the gesture.

        Returns:
            str: The name of the gesture (e.g., "fist", "palm_up"), or "unknown"
                 if the gesture ID is not found or invalid.
        """
        if gesture_id == -1:
            return "unknown"
        key_binding = self.get_key_bindings(gesture_id)
        return key_binding.get("gesture", "unknown")

    def get_keys_for_gesture(self, gesture_id: int) -> List[str]:
        """
        Retrieves the list of keyboard keys to be pressed for a given gesture ID.

        Args:
            gesture_id (int): The ID of the gesture.

        Returns:
            List[str]: A list of keys (e.g., ["w", "a"]), or an empty list if the
                       gesture is not found or has no keys assigned.
        """
        key_binding = self.get_key_bindings(gesture_id)
        return key_binding.get("keys", [])

    def get_behavior_for_gesture(self, gesture_id: int) -> Optional[str]:
        """
        Retrieves the press behavior for a given gesture's keys ('press' or 'hold').

        Args:
            gesture_id (int): The ID of the gesture.

        Returns:
            Optional[str]: The key press behavior as a string, or None if not defined.
        """
        key_binding = self.get_key_bindings(gesture_id)
        return key_binding.get("behavior")

    def get_all_keys(self) -> List[str]:
        """
        Retrieves a unique list of all keys used across all gesture bindings.

        Returns:
            List[str]: A list of all unique keyboard keys defined in the key bindings.
        """
        all_keys: Set[str] = set()
        for gesture_data in self.key_bindings_data.values():
            keys = gesture_data.get("keys", [])
            all_keys.update(keys)
        return list(all_keys)

    def get_all_gesture_ids(self) -> List[int]:
        """
        Retrieves a list of all configured gesture IDs.

        Returns:
            List[int]: A list of all gesture IDs from the key bindings configuration.
        """
        return [int(gesture_id) for gesture_id in self.key_bindings_data.keys()]

    def get_hand_gesture_config(self) -> Dict[str, Any]:
        """
        Retrieves the configuration section for hand gesture detection.

        Returns:
            Dict[str, Any]: A dictionary of settings for the gesture detector,
                            such as confidence thresholds and detection rates.
        """
        return self.get_app_config('HAND_GESTURE_CONFIG')

    def get_neural_network_config(self) -> Dict[str, Any]:
        """
        Retrieves the configuration section for the neural network.

        Returns:
            Dict[str, Any]: A dictionary of settings for the neural network,
                            such as the model path and dataset path.
        """
        return self.get_app_config('NEURAL_NETWORK_CONFIG')

    def get_hyperparameter_config(self) -> Dict[str, Any]:
        """
        Retrieves the configuration section for hyperparameter tuning.

        Returns:
            Dict[str, Any]: A dictionary defining the search space for
                            hyperparameters like learning rate and batch size.
        """
        return self.get_app_config('HYPERPARAMETER_CONFIG')
