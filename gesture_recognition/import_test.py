import cv2
import mediapipe as mp
from background_subtraction import take_raw_frame_and_convert_to_contours
import numpy as np


def main():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        hand_outputs = take_raw_frame_and_convert_to_contours(frame)
        if hand_outputs not in [None, []]:
            cv2.imshow("MediaPipe Hands", hand_outputs[0]["mask"])
        else:
            cv2.imshow("MediaPipe Hands", np.zeros((240, 320, 3), dtype=np.uint8))

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()


if __name__ == "__main__":

    main()
