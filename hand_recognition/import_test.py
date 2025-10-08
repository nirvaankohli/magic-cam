import cv2
import mediapipe as mp
import recognitionv2 as rec


def main():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        frame = rec.process_frame(cv2.flip(frame, 1))

        cv2.imshow("MediaPipe Hands", frame)

        


if __name__ == "__main__":

    main()
