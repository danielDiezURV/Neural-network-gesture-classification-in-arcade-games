from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from src.gesture_controller.app_config import AppConfig
import tensorflow as tf
import numpy as np
import os

class GestureClassifier:

    def __init__(self, num_classes=None, input_size=None, hyperparams=None):
        self.num_classes = num_classes
        self.input_size = input_size
        
        nn_config = AppConfig().get_neural_network_config()
        self.model_path = nn_config.get('MODEL_PATH')

        if hyperparams:
            self.model = self._create_model(
                layers=hyperparams['DENSE_LAYERS'],
                activation=hyperparams['ACTIVATION'],
                dropout_rate=hyperparams['DROPOUT_RATE'],
                learning_rate=hyperparams['LEARNING_RATE']
            )
        else:
            self.model = None
        

    # Creates the neural network model architecture with sparse categorical crossentropy.
    #
    # The model consists of:
    # - Dense layers with ReLU activation for feature extraction
    # - Dropout layers for regularization to prevent overfitting
    # - Final softmax layer for multi-class classification
    #
    # Returns:
    #     Sequential: A compiled Keras sequential model ready for training.
    def _create_model(self, layers, activation, dropout_rate, learning_rate):
        model = Sequential()
        # Use an explicit Input layer to avoid passing input_shape to Dense
        model.add(Input(shape=(self.input_size,)))
        model.add(Dense(layers[0], activation=activation))
        
        for units in layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(dropout_rate))

        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    # Loads a pre-trained model from the specified model path.
    #
    # Returns:
    #     bool: True if model loaded successfully, False otherwise.
    def load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
        

    # Trains the gesture classification model using the provided training and validation data.
    #
    # The training process includes:
    # - Model architecture creation and compilation
    # - Training with specified epochs and batch size
    # - Model saving to the configured path
    # - Performance evaluation on both training and validation sets
    #
    # Args:
    #     X_train (np.ndarray): Training feature data with shape (n_samples, 60).
    #     Y_train (np.ndarray): Training labels as 0-indexed integers.
    #     X_val (np.ndarray): Validation feature data with shape (n_samples, 60).
    #     Y_val (np.ndarray): Validation labels as 0-indexed integers.
    #
    # Returns:
    #     History: Keras training history object containing loss and accuracy metrics.
    def train(self, X_train, Y_train, X_val, Y_val, batch_size, epochs):
        self.model.summary()

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        history = self.model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            verbose=1
        )
        _, train_acc = self.model.evaluate(X_train, Y_train, verbose=0)
        _, val_acc = self.model.evaluate(X_val, Y_val, verbose=0)
        
        print(f"\nFinal Training Accuracy: {train_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        
        return self.model, history

    
    # Predicts gesture IDs for the given dataset.
    #
    # Args:
    #     X_data (np.ndarray): Input data with shape (n_samples, n_features).
    #
    # Returns:
    #     np.ndarray: Predicted gesture IDs.
    def evaluate(self, X_data):
        probabilities = self.model.predict(X_data, verbose=0)
        return np.argmax(probabilities, axis=1)
    

    # Predicts gesture from preprocessed landmark vector.
    #
    # The prediction process involves:
    # - Input validation and reshaping
    # - Model inference to get class probabilities
    # - Converting 0-indexed predictions back to original gesture IDs
    #
    # Args:
    #     landmark_vector (np.ndarray or None): Preprocessed landmark vector with 60 elements.
    #
    # Returns:
    #     int: The predicted gesture ID 
    def predict(self, landmark_vector):
        if self.model is None:
            return -1
        
        if landmark_vector is None:
            return -1
        
        input_data = np.array(landmark_vector).reshape(1, -1)
        
        predictions = self.model.predict(input_data, verbose=0)
        
        predicted_class = np.argmax(predictions[0])

        return predicted_class
        