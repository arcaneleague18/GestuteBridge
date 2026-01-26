import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(actions_shape):
    model = Sequential()
    # input_shape = (sequence_length, features_per_frame)
    # We used 30 frames, and 126 features (2 hands * 21 landmarks * 3 coords)
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions_shape, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def load_action_model(actions_shape, model_paths=None):
    """
    Attempt to load the serialized model first. If that fails (e.g. legacy H5 with time_major),
    rebuild the architecture and load weights only.
    """
    if model_paths is None:
        model_paths = (
            os.path.join(ROOT_DIR, "action.keras"),
            os.path.join(ROOT_DIR, "action.h5"),
        )

    found_artifact = False
    last_error = None

    for path in model_paths:
        if not path:
            continue
        if not os.path.exists(path):
            continue

        found_artifact = True
        try:
            return keras.models.load_model(path, compile=False, safe_mode=False)
        except Exception as exc:
            last_error = exc

            if path.lower().endswith('.h5'):
                try:
                    legacy_model = build_model(actions_shape)
                    legacy_model.load_weights(path)
                    print(f"Loaded legacy weights from '{path}'.")
                    return legacy_model
                except Exception as weight_exc:
                    last_error = weight_exc

    if not found_artifact:
        raise FileNotFoundError(
            f"No model artifacts found. Expected one of: {', '.join(model_paths)}"
        )

    raise RuntimeError(f"Model artifacts exist but could not be loaded. Last error: {last_error}")
