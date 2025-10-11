import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import os

model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
)

with open(model_path, "rb") as f:
    model_bytes = f.read()

base_options = python.BaseOptions(model_asset_buffer=model_bytes)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


def recognize_gesture(image):

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize(mp_image)

    return recognition_result


def draw_results(recognition_result, frame, text=True, landmarks=True):

    top_gesture = recognition_result.gestures[0][0]
    gesture_name = top_gesture.category_name
    confidence = top_gesture.score

    img = frame.copy()

    if text:

        cv2.putText(
            img,
            f"Gesture: {gesture_name} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if recognition_result.hand_landmarks:
        for hand_landmarks in recognition_result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    return img


def main():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        recognition_result = recognize_gesture(frame)

        if recognition_result.gestures:

            frame = draw_results(recognition_result, frame)

        cv2.imshow("MediaPipe Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


if __name__ == "__main__":

    main()
