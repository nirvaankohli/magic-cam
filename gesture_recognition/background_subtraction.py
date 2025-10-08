import cv2
import mediapipe.python.solutions.hands as mp_hands
from pathlib import Path
import sys


module_dir = Path(__file__).parent + "/hand_recognition"
sys.path.insert(0, str(module_dir))


def main():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("Magic Cam - Only Hand Recognition - press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
