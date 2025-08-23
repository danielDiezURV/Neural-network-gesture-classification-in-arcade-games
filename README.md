# Neural Network Gesture Classification for Arcade Games

Turn hand gestures into keystrokes to control arcade games. This project detects hands with MediaPipe, preprocesses 3D landmarks, classifies gestures with a neural network, and maps them to keyboard inputs in real time.

## Features
- Real-time hand tracking with MediaPipe and OpenCV
- Robust preprocessing (translation/scale invariant)
- Configurable neural network with hyperparameter tuning
- Pluggable key bindings per gesture
- Training notebook with best-model selection and model export

## Project structure
- `main.py` — Real-time app: webcam → gesture → keystrokes
- `gesture_detector.py` — MediaPipe hand detection and landmark extraction
- `data_preprocessor.py` — Landmark normalization and flattening (60 features)
- `gesture_classifier.py` — Keras classifier (Dense + Dropout, Softmax)
- `keyboard_adapter.py` — Maps gesture IDs to keys and presses/releases them
- `config/app_config.json` — App, model, and hyperparameter config
- `data/gestures_dataset.csv` — Labeled dataset of preprocessed landmarks
- `data/key_bindings_default.json` — Default mapping from gesture IDs to keys
- `neural_network_training.ipynb` — Dataset load, tuning, and model export
- `models/gesture_model.h5` — Trained model (saved by the training step)

## Requirements
- macOS, Linux, or Windows
- Python 3.10+ (3.11 recommended)
- Webcam access (grant camera permission on macOS)

## Setup
Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If Jupyter isn’t available:

```bash
pip install jupyter
```

## Configuration
Edit `config/app_config.json`:

- `HAND_GESTURE_CONFIG`
  - `MAX_HANDS`: 1–2 hands to track
  - `MIN_DETECTION_CONFIDENCE`: 0.0–1.0 threshold (e.g., 0.7)
  - `DETECTION_RATE`: seconds between classification updates
  - `DEFAULT_KEY_BINDINGS`: path to key bindings JSON (e.g., `data/key_bindings_default.json`)
- `NEURAL_NETWORK_CONFIG`
  - `MODEL_PATH`: where the trained model is saved/loaded (e.g., `models/gesture_model.h5`)
  - `DATASET_PATH`: CSV created by data generation (e.g., `data/gestures_dataset.csv`)
- `HYPERPARAMETER_CONFIG` (used by the training notebook)
  - `DENSE_LAYERS`: e.g., `[[128, 64, 32], [256, 128, 64]]`
  - `ACTIVATION`: e.g., `["relu", "tanh", "sigmoid"]`
  - `DROPOUT_RATE`: e.g., `[0.2, 0.4]`
  - `LEARNING_RATE`: e.g., `[0.01, 0.001]`
  - `BATCH_SIZE`: e.g., `[32, 64]`
  - `EPOCHS`: e.g., `[100]`

Key bindings file (`data/key_bindings_default.json`) example entry:

```json
{
  "0": { "gesture": "palm_up",    "keys": ["SPACE"], "behavior": "tap" },
  "1": { "gesture": "palm_down",  "keys": ["UP"],    "behavior": "hold" },
  "2": { "gesture": "fist",       "keys": ["LEFT"],  "behavior": "hold" }
}
```

- `gesture`: human-friendly name
- `keys`: list of keys to press
- `behavior`: "tap" (press+release) or "hold" (press until gesture changes)

## How it works
1. Capture frames with OpenCV; mirror image for natural interaction.
2. Detect 21 3D hand landmarks (MediaPipe Hands).
3. Preprocess landmarks (`data_preprocessor.py`):
   - Center on wrist (translation invariant)
   - Scale by max distance (scale invariant)
   - Remove wrist landmark
   - Flatten to a 60‑dim vector (20 landmarks × xyz)
4. Classify with a Keras Sequential model (`gesture_classifier.py`):
   - Input(60) → Dense/Dropout stacks → Softmax(num_classes)
   - Loss: `sparse_categorical_crossentropy`; Optimizer: Adam (configurable)
5. Map predicted gesture ID to configured keys (`keyboard_adapter.py`).

## Train the model (notebook)
Open `neural_network_training.ipynb` and run the cells:

- Loads `DATASET_PATH` and splits into train/val
- Grid‑searches hyperparameters from `HYPERPARAMETER_CONFIG`
- Trains each variant and records `val_accuracy` and `val_loss`
- Selects the best run by highest `val_accuracy` (tie‑broken by lower `val_loss`)
- Saves the best model to `MODEL_PATH` (e.g., `models/gesture_model.h5`)

Notes
- Dataset column `LANDMARKS` must be a list‑like of 60 floats; `GESTURE_ID` are 0‑indexed class IDs.
- If you don’t have a dataset, see `data/data_generation.ipynb` to create one from images or live capture.

## Run the real‑time app
Ensure your trained model exists at `MODEL_PATH`.

```bash
python main.py
```

Shortcuts
- ESC to quit the OpenCV window
- The terminal logs gesture changes and corresponding key actions

macOS tips
- Grant Terminal/VS Code camera access (System Settings → Privacy & Security → Camera)
- If you see a black window, try reducing resolution or closing other apps using the camera

## Model and tuning details
- Input size: 60 features (normalized landmarks without wrist)
- Output: `num_classes` (derived from max label in the dataset)
- Architecture: Dense → Dropout stacks defined by `DENSE_LAYERS`
- Activation: configurable (e.g., relu/tanh/sigmoid)
- Optimizers: Adam by default; you can also use SGD(momentum), RMSprop, AdamW, Nadam
- Training metrics: `accuracy` and `val_accuracy` with companion losses

To try a different optimizer, extend the model compile step in `gesture_classifier.py` or add an `OPTIMIZER` option to your hyperparameter grid and branch on its value before `model.compile`.

## Add new gestures
1. Collect images or capture sequences for the new gesture(s)
2. Generate/update `gestures_dataset.csv` (see `data/data_generation.ipynb`)
3. Update key bindings with new gesture IDs and names
4. Retrain with the notebook; the best model will be saved to `MODEL_PATH`

## Troubleshooting
- Python not found: Install Python 3.11+, then recreate the venv and reinstall deps
- Camera not available: Close other apps using the camera; grant permissions
- Model file missing: Train via the notebook to create `models/gesture_model.h5`
- Dataset not found: Confirm `DATASET_PATH` in `config/app_config.json`; regenerate if needed
- Low accuracy: Increase dataset size/diversity; tune layers, dropout, learning rate, and epochs

## Disclaimer
Use responsibly; key injection may require accessibility permissions depending on OS.
