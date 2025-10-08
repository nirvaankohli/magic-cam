import cv2 

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

def main():

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(

        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():

            success, iframe = cap.read()

            if not success:

                print("Ignoring empty camera frame.")

                continue

            cv2.imshow("Magic Cam - Only Hand Recognition - press 'q' to quit", iframe)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break

    cap.release()

    