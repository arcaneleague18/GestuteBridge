import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from model import build_model, load_action_model
from actions import get_actions, save_actions, load_actions

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "MP_Data")
MODEL_KERAS_PATH = os.path.join(ROOT_DIR, "action.keras")
MODEL_H5_PATH = os.path.join(ROOT_DIR, "action.h5")
ACTION_LABELS_PATH = os.path.join(ROOT_DIR, "actions.txt")
LOGS_PATH = os.path.join(ROOT_DIR, "logs")
sequence_length = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune sign model.")
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "finetune"],
        default="auto",
        help="Training mode. auto picks finetune when compatible model+labels exist.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for full training.")
    parser.add_argument("--finetune-epochs", type=int, default=100, help="Epochs for fine-tuning.")
    parser.add_argument(
        "--focus-action",
        type=str,
        default="",
        help="Focus training on one sign (e.g., SWAG) while replaying a small subset of other signs.",
    )
    parser.add_argument(
        "--replay-per-other",
        type=int,
        default=5,
        help="When --focus-action is set, number of sequences to include from each non-focused sign.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard callback (off by default for better backend compatibility).",
    )
    return parser.parse_args()

def list_sequence_dirs(action):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        return []
    return sorted(
        int(name)
        for name in os.listdir(action_path)
        if os.path.isdir(os.path.join(action_path, name)) and name.isdigit()
    )

def load_sequence(action, sequence):
    window = []
    for frame_num in range(sequence_length):
        frame_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
        if not os.path.exists(frame_path):
            return None
        window.append(np.load(frame_path))
    return window

def load_dataset(actions, focus_action="", replay_per_other=0):
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    print(f"Using actions: {', '.join(actions)}")
    if focus_action:
        print(f"Focused action: {focus_action} (replay per other action: {replay_per_other})")

    for action in actions:
        sequence_dirs = list_sequence_dirs(action)
        if focus_action:
            if action == focus_action:
                selected_sequences = sequence_dirs
            else:
                selected_sequences = sequence_dirs[: max(0, replay_per_other)]
        else:
            selected_sequences = sequence_dirs

        for sequence in selected_sequences:
            window = load_sequence(action, sequence)
            if window is not None:
                sequences.append(window)
                labels.append(label_map[action])

    if not sequences:
        raise ValueError("No complete sequences found in MP_Data.")

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    return X, y

def get_compatible_finetune_model(actions):
    if not os.path.exists(MODEL_KERAS_PATH) and not os.path.exists(MODEL_H5_PATH):
        return None, "no saved model found"

    saved_actions = load_actions(file_path=ACTION_LABELS_PATH, data_path=DATA_PATH)
    if saved_actions != list(actions):
        return None, "saved actions do not match current MP_Data actions"

    try:
        model = load_action_model(
            actions_shape=len(actions),
            model_paths=(MODEL_KERAS_PATH, MODEL_H5_PATH),
        )
    except Exception as exc:
        return None, f"existing model is incompatible for fine-tuning ({exc})"
    output_classes = model.output_shape[-1]
    if output_classes != len(actions):
        return None, "model output size does not match current actions"
    return model, None

def build_warm_started_model(actions, saved_actions, saved_model):
    new_model = build_model(len(actions))

    for layer_idx in range(len(new_model.layers) - 1):
        if layer_idx >= len(saved_model.layers) - 1:
            break
        new_model.layers[layer_idx].set_weights(saved_model.layers[layer_idx].get_weights())

    old_kernel, old_bias = saved_model.layers[-1].get_weights()
    new_kernel, new_bias = new_model.layers[-1].get_weights()

    old_index = {label: idx for idx, label in enumerate(saved_actions)}
    for new_idx, label in enumerate(actions):
        old_idx = old_index.get(label)
        if old_idx is None:
            continue
        new_kernel[:, new_idx] = old_kernel[:, old_idx]
        new_bias[new_idx] = old_bias[old_idx]

    new_model.layers[-1].set_weights([new_kernel, new_bias])
    return new_model

def get_transfer_model(actions):
    if not os.path.exists(MODEL_KERAS_PATH) and not os.path.exists(MODEL_H5_PATH):
        return None, "no saved model found"
    if not os.path.exists(ACTION_LABELS_PATH):
        return None, "actions.txt not found for previous model labels"

    saved_actions = load_actions(file_path=ACTION_LABELS_PATH, data_path=DATA_PATH)
    try:
        saved_model = load_action_model(
            actions_shape=len(saved_actions),
            model_paths=(MODEL_KERAS_PATH, MODEL_H5_PATH),
        )
    except Exception as exc:
        return None, f"could not load previous model for transfer ({exc})"

    return build_warm_started_model(actions, saved_actions, saved_model), None

def build_callbacks(enable_tensorboard):
    if not enable_tensorboard:
        return []
    try:
        return [keras.callbacks.TensorBoard(log_dir=LOGS_PATH)]
    except Exception as exc:
        print(f"TensorBoard callback unavailable: {exc}. Continuing without TensorBoard logs.")
        return []

def train_model(args):
    actions = np.array(get_actions(DATA_PATH))
    focus_action = args.focus_action.strip().upper()
    if focus_action and focus_action not in actions:
        raise ValueError(f"Focused action '{focus_action}' is not present in MP_Data.")

    X, y = load_dataset(actions, focus_action=focus_action, replay_per_other=args.replay_per_other)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

    model = None
    mode = args.mode
    if args.mode in {"auto", "finetune"}:
        model, reason = get_compatible_finetune_model(actions)
        if model is not None:
            mode = "finetune"
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy"],
            )
            print(f"Fine-tuning existing model for {args.finetune_epochs} epochs.")
        elif args.mode == "finetune":
            raise RuntimeError(f"Cannot fine-tune: {reason}")
        else:
            print(f"Auto mode selected full training: {reason}.")

    if model is None:
        if focus_action:
            model, transfer_reason = get_transfer_model(actions)
            if model is not None:
                mode = "focused"
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss="categorical_crossentropy",
                    metrics=["categorical_accuracy"],
                )
                print(f"Using transfer warm-start for focused training on '{focus_action}'.")
            else:
                print(f"Focused transfer unavailable: {transfer_reason}.")

    if model is None:
        mode = "full"
        model = build_model(actions.shape[0])
        print(f"Training new model for {args.epochs} epochs.")

    if mode in {"finetune", "focused"}:
        epochs = args.finetune_epochs
    else:
        epochs = args.epochs
    callbacks = build_callbacks(args.tensorboard)
    model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks)
    model.save(MODEL_KERAS_PATH)
    save_actions(actions.tolist(), file_path=ACTION_LABELS_PATH)
    print(f"Model saved as {MODEL_KERAS_PATH}")
    print(f"Action labels saved to {ACTION_LABELS_PATH}")

if __name__ == "__main__":
    print("Loading data...")
    args = parse_args()
    try:
        train_model(args)
    except Exception as e:
        print(f"Error loading data or training: {e}")
        print("Make sure you have run data_collection.py first to generate data.")
