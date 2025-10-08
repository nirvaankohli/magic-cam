import cv2
import mediapipe.python.solutions.hands as mp_hands
from pathlib import Path
import sys

module_dir = Path(__file__).parent.parent / "hand_recognition"
sys.path.insert(0, str(module_dir))


def get_hand_bboxes(multi_hand_landmarks, img_width, img_height):

    bboxes = []

    if multi_hand_landmarks:

        for hand_landmarks in multi_hand_landmarks:

            x_coords = [lm.x * img_width for lm in hand_landmarks.landmark]
            y_coords = [lm.y * img_height for lm in hand_landmarks.landmark]

            padding = 10

            x_min = max(int(min(x_coords)) - padding, 0)
            x_max = min(int(max(x_coords)) + padding, img_width)
            y_min = max(int(min(y_coords)) - padding, 0)
            y_max = min(int(max(y_coords)) + padding, img_height)

            bboxes.append((x_min, y_min, x_max, y_max))

    return bboxes


import recognitionv2 as hrec


def process_frame_no_landmarks(frame, hands):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    return frame, results


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

        fr, results = process_frame_no_landmarks(frame, hands)

        if results.multi_hand_landmarks:

            hand_bboxes = get_hand_bboxes(
                results.multi_hand_landmarks, frame.shape[1], frame.shape[0]
            )

            display_frame = frame.copy()
            for bbox in hand_bboxes:
                display_frame = cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (255, 0, 0),
                    2,
                )

        else:

            display_frame = frame

        cv2.imshow(
            "Magic Cam - Only Hand Recognition - press 'q' to quit", display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
