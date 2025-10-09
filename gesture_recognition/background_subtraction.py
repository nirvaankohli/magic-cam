import cv2
import numpy as np
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

            padding = 30

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


def get_hand_number_landmarks():

    return (
        0,
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    )


def convert_landmarks_to_cropped(
    hand_landmarks, cropped_image, img_width, img_height, bbox
):

    hand_points = []

    for lm in hand_landmarks.landmark:

        x = int(lm.x * img_width)
        y = int(lm.y * img_height)
        x_crop = np.clip(x - bbox[0], 0, cropped_image.shape[1] - 1)
        y_crop = np.clip(y - bbox[1], 0, cropped_image.shape[0] - 1)
        hand_points.append([x_crop, y_crop])

    hand_points = np.array(hand_points, dtype=np.int32)

    return hand_points


def apply_after_mask_effects(mask):

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.dilate(mask, kernel_dilate, iterations=2)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def create_hand_mask(cropped_image, hand_landmarks, bbox, img_width, img_height):

    mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    x_min, y_min, x_max, y_max = bbox

    WRIST, THUMB, INDEX, MIDDLE, RING, PINKY = get_hand_number_landmarks()

    hand_points = convert_landmarks_to_cropped(
        hand_landmarks, cropped_image, img_width, img_height, bbox
    )

    ordered_contour = []

    ordered_contour.extend(
        [hand_points[i] for i in [0, 1, 2, 3, 4]]
    )  # Wrist to thumb tip
    ordered_contour.extend(
        [hand_points[i] for i in [8, 7, 6, 5]]
    )  # To index tip and back
    ordered_contour.extend(
        [hand_points[i] for i in [9, 10, 11, 12, 11, 10, 9]]
    )  # To middle tip and back
    ordered_contour.extend(
        [hand_points[i] for i in [13, 14, 15, 16, 15, 14, 13]]
    )  # To ring tip and back
    ordered_contour.extend(
        [hand_points[i] for i in [17, 18, 19, 20, 19, 18, 17, 0]]
    )  # To pinky tip and back to wrist

    # Convex hull :D
    hull = cv2.convexHull(hand_points)
    cv2.fillPoly(mask, [hull], 255)

    # Add detailed finger regions using individual finger polygons

    for finger_indices in [THUMB, INDEX, MIDDLE, RING, PINKY]:

        finger_points = hand_points[finger_indices]

        palm_base = hand_points[[0, 5, 9, 13, 17]]
        extended_finger = np.vstack([finger_points, palm_base[np.newaxis, 0]])
        finger_hull = cv2.convexHull(extended_finger)

        cv2.fillPoly(mask, [finger_hull], 255)

    mask = apply_after_mask_effects(mask)

    return mask


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

            for i, (bbox, hand_landmarks) in enumerate(
                zip(hand_bboxes, results.multi_hand_landmarks)
            ):

                # Draw bounding box
                display_frame = cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (255, 0, 0),
                    2,
                )

                x_min, y_min, x_max, y_max = bbox
                cropped_hand = frame[y_min:y_max, x_min:x_max]

                if cropped_hand.size > 0:

                    hand_mask = create_hand_mask(
                        cropped_hand,
                        hand_landmarks,
                        bbox,
                        frame.shape[1],
                        frame.shape[0],
                    )

                    # Convert single channel mask to 3 channel for overlay
                    hand_mask_3ch = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)

                    # Replace the cropped region in display_frame with the mask
                    display_frame[y_min:y_max, x_min:x_max] = hand_mask_3ch

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
