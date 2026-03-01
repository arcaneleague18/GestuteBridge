import cv2
import numpy as np
import os
from actions import load_actions

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "MP_Data")
ACTIONS = set(load_actions(data_path=DATA_PATH))

def draw_landmarks(image, landmarks_flat):
    h, w, c = image.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    hand1_data = landmarks_flat[:63]
    if np.any(hand1_data):
        draw_hand(image, hand1_data, connections, (255, 0, 0))

    hand2_data = landmarks_flat[63:]
    if np.any(hand2_data):
        draw_hand(image, hand2_data, connections, (0, 0, 255))

def draw_hand(image, hand_data, connections, color):
    h, w, c = image.shape
    points = []
    for i in range(21):
        x = hand_data[i*3]
        y = hand_data[i*3 + 1]
        if x == 0.0 and y == 0.0:
            points.append(None)
        else:
            cx, cy = int(x * w), int(y * h)
            points.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)

    for p1_idx, p2_idx in connections:
        if points[p1_idx] is not None and points[p2_idx] is not None:
            cv2.line(image, points[p1_idx], points[p2_idx], (255, 255, 255), 2)

def expand_units(raw_text):
    normalized = raw_text.strip().upper()
    if not normalized:
        return []
    if normalized in ACTIONS:
        return [normalized]

    units = []
    words = normalized.split()
    for idx, word in enumerate(words):
        if word in ACTIONS:
            units.append(word)
        else:
            units.extend(list(word))
        if idx < len(words) - 1:
            units.append(" ")
    return units

def get_sequence_frames(action):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        return []

    sequence_dirs = sorted(
        int(name)
        for name in os.listdir(action_path)
        if os.path.isdir(os.path.join(action_path, name)) and name.isdigit()
    )
    if not sequence_dirs:
        return []

    sequence_path = os.path.join(action_path, str(sequence_dirs[0]))
    frame_files = sorted(
        int(os.path.splitext(name)[0])
        for name in os.listdir(sequence_path)
        if name.endswith(".npy") and os.path.splitext(name)[0].isdigit()
    )
    return [os.path.join(sequence_path, f"{frame}.npy") for frame in frame_files]

def play_sign_for_text(text):
    units = expand_units(text)
    if not units:
        return
    
    for unit in units:
        if unit == " ":
            print("Space")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "SPACE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Reverse Translation', blank)
            cv2.waitKey(1000)
            continue

        frame_paths = get_sequence_frames(unit)
        if not frame_paths:
            print(f"No data found for sign: {unit}")
            continue

        print(f"Showing sign for: {unit}")
        
        for npy_path in frame_paths:
            landmarks = np.load(npy_path)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, f"Translating: {text.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Current: {unit}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            draw_landmarks(image, landmarks)
            cv2.imshow('Reverse Translation', image)
            key = cv2.waitKey(33)
            if key == ord('q'):
                return

        cv2.waitKey(200)

if __name__ == "__main__":
    print("Loaded signs:", ", ".join(sorted(ACTIONS)))
    while True:
        user_input = input("Enter text to translate (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        play_sign_for_text(user_input)
    
    cv2.destroyAllWindows()
