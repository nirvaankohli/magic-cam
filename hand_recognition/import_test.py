import cv2
import mediapipe as mp


def main():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        frame = cv2.flip(frame, 1)


if __name__ == "__main__":

    main()
