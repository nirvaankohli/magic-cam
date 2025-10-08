import cv2
import mediapipe.python.solutions.hands as mp_hands
from pathlib import Path
import sys

module_dir = Path(__file__).parent.parent / "hand_recognition"
sys.path.insert(0, str(module_dir))


def get_hand_bbox(lm, img_width, img_height):

    x_coords = [lm.x * img_width for lm in lm.landmark]
    y_coords = [lm.y * img_height for lm in lm.landmark]


import recognitionv2 as hrec


def main():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        fr, results = hrec.process_frame(frame, hands)

        print(
            get_hand_bbox(
                results.multi_hand_landmarks[0], frame.shape[1], frame.shape[0]
            )
            if results.multi_hand_landmarks
            else "No hands detected"
        )

        cv2.imshow("Magic Cam - Only Hand Recognition - press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
