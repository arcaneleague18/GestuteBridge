import cv2
import numpy as np
import os
from hand_tracking import HandTracker

import string
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "MP_Data")
DEFAULT_ACTIONS = list(string.ascii_uppercase)
DEFAULT_NO_SEQUENCES = 30
DEFAULT_SEQUENCE_LENGTH = 30

def list_existing_actions():
    if not os.path.exists(DATA_PATH):
        return []
    return sorted(
        action for action in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, action))
    )

def parse_actions(raw_actions):
    actions = []
    for token in raw_actions.split(","):
        token = token.strip().upper()
        if token and token not in actions:
            actions.append(token)
    return actions

def get_next_sequence_index(action):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        return 0

    numeric_sequences = [
        int(name)
        for name in os.listdir(action_path)
        if os.path.isdir(os.path.join(action_path, name)) and name.isdigit()
    ]
    return max(numeric_sequences) + 1 if numeric_sequences else 0

def prompt_int(prompt, default_value):
    raw = input(f"{prompt} [{default_value}]: ").strip()
    if not raw:
        return default_value
    value = int(raw)
    if value <= 0:
        raise ValueError(f"Expected a positive integer for '{prompt}'.")
    return value

def select_actions():
    existing_actions = list_existing_actions()
    print("Choose data collection mode:")
    print("1) Append to existing signs")
    print("2) Add new signs")
    print("3) Custom sign list (append existing + create new)")
    print("4) Use default A-Z")

    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice in {"1", "2", "3", "4"}:
            break
        print("Invalid choice. Enter 1, 2, 3, or 4.")

    if choice == "4":
        return DEFAULT_ACTIONS

    if choice == "1":
        if not existing_actions:
            print("No existing signs found in MP_Data. Switching to custom mode.")
            choice = "3"
        else:
            print(f"Existing signs: {', '.join(existing_actions)}")
            raw = input("Enter signs to append (comma-separated, blank for all): ").strip()
            selected = existing_actions if not raw else parse_actions(raw)
            invalid = [action for action in selected if action not in existing_actions]
            if invalid:
                raise ValueError(f"These signs do not exist yet: {', '.join(invalid)}")
            return selected

    if choice == "2":
        raw = input("Enter new signs to add (comma-separated): ").strip()
        selected = parse_actions(raw)
        if not selected:
            raise ValueError("No signs provided.")
        return selected

    raw = input("Enter signs to collect (comma-separated): ").strip()
    selected = parse_actions(raw)
    if not selected:
        raise ValueError("No signs provided.")
    return selected

if __name__ == "__main__":
    # Initialize HandTracker
    tracker = HandTracker(max_hands=2)
    os.makedirs(DATA_PATH, exist_ok=True)

    actions = np.array(select_actions())
    no_sequences = prompt_int("Number of new sequences per sign", DEFAULT_NO_SEQUENCES)
    sequence_length = prompt_int("Frames per sequence", DEFAULT_SEQUENCE_LENGTH)

    cap = cv2.VideoCapture(0)

    print("Starting Data Collection...")
    print("Controls: SPACE = start sign, S = skip sign, Q = quit.")

    stop_all = False
    for action in actions:
        existing_actions = list_existing_actions()
        start_sequence = get_next_sequence_index(action)
        mode = "Appending" if action in existing_actions else "Creating"
        print(f"{mode} data for '{action}' starting at sequence {start_sequence}")

        skip_action = False
        while True:
            ret, frame = cap.read()
            if not ret:
                stop_all = True
                break

            cv2.putText(frame, f'Collecting {action}. Press "SPACE" to start.', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            key = cv2.waitKey(10) & 0xFF
            if key == 32:
                break
            if key == ord('s'):
                skip_action = True
                break
            if key == ord('q'):
                stop_all = True
                break

        if stop_all:
            break
        if skip_action:
            continue

        for sequence_offset in range(no_sequences):
            sequence = start_sequence + sequence_offset
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))
            while os.path.exists(sequence_path):
                sequence += 1
                sequence_path = os.path.join(DATA_PATH, action, str(sequence))

            os.makedirs(sequence_path, exist_ok=False)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    stop_all = True
                    break

                image = tracker.find_hands(frame)
                keypoints = tracker.find_position(frame)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                cv2.putText(image, f'Collecting {action} Sequence {sequence}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop_all = True
                    break

            if stop_all:
                break

    cap.release()
    cv2.destroyAllWindows()

