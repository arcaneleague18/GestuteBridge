import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Import new Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(ROOT_DIR, "hand_landmarker.task")

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        # Convert the image to MediaPipe's Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Detect hands (synchronous)
        self.results = self.landmarker.detect(mp_image)

        if self.results.hand_landmarks:
            if draw:
                # Reuse the drawing utils from the old API (it still exists usually as a utility)
                # If mp.solutions.drawing_utils exists, else implement custom drawing
                # To be safe given the user error, let's implement basic manual drawing if solutions is missing
                try:
                    mp_draw = mp.solutions.drawing_utils
                    mp_hands_style = mp.solutions.hands
                    for hand_landmarks in self.results.hand_landmarks:
                        # Convert hand_landmarks (NormalizedLandmark) to the format expected by drawing_utils
                        # The new result is a list of lists of NormalizedLandmark objects
                        # drawing_utils expects a NormalizedLandmarkList proto
                        import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
                        landmark_list = landmark_pb2.NormalizedLandmarkList()
                        landmark_list.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                            for landmark in hand_landmarks
                        ])
                        mp_draw.draw_landmarks(img, landmark_list, mp_hands_style.HAND_CONNECTIONS)
                except AttributeError:
                    # Fallback manual drawing
                    h, w, c = img.shape
                    for hand_landmarks in self.results.hand_landmarks:
                        for landmark in hand_landmarks:
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return img

    def find_position(self, img):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            hands = sorted(
                self.results.hand_landmarks,
                key=lambda hand_landmarks: hand_landmarks[0].x if hand_landmarks else 0.0,
            )
            for hand_landmarks in hands:
                for lm in hand_landmarks:
                    lm_list.extend([lm.x, lm.y, lm.z])
            
            # Padding - we expect 126 features (2 hands * 21 landmarks * 3 coords)
            # If only 1 hand detected (63 features), we pad with zeros
            expected_len = 2 * 21 * 3 
            if len(lm_list) < expected_len:
                lm_list.extend([0.0] * (expected_len - len(lm_list)))
            elif len(lm_list) > expected_len:
                lm_list = lm_list[:expected_len]
        else:
            lm_list = [0.0] * (2 * 21 * 3)

        return lm_list
