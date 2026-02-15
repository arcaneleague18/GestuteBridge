import os
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import pathlib
from model import load_action_model
from hand_tracking import HandTracker
import keras
from actions import load_actions

ROOT_DIR = pathlib.Path(__file__).resolve().parent
actions = np.array(load_actions(data_path=str(ROOT_DIR / "MP_Data")))

# Load model
# Prefer modern .keras archives, but fall back to legacy .h5 weights if needed.
print(f"Keras version: {keras.__version__}")
try:
    model = load_action_model(actions_shape=actions.shape[0])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("No usable model weights found. Please run train.py first.")
    exit(1)
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
tracker = HandTracker(max_hands=2)

print("Starting Inference...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    image = tracker.find_hands(frame)
    keypoints = tracker.find_position(frame)
    
    sequence.append(keypoints)
    sequence = sequence[-30:] # Keep last 30 frames
    
    if len(sequence) % 10 == 0:
        print(f"Sequence length: {len(sequence)}")

    if len(sequence) == 30:
        try:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Visualization logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 2: 
                sentence = sentence[-2:]
            
            # Use safe font or just basic
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except Exception as e:
            # Model might not be loaded
            print(f"Error during prediction: {e}")
            pass

    cv2.imshow('OpenCV Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
