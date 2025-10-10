import cv2
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
from pathlib import Path
import sys

module_dir = Path(__file__).parent.parent / "hand_recognition"
sys.path.insert(0, str(module_dir))


def take_raw_frame_and_convert_to_contours(frame):

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_outputs = []

    if results.multi_hand_landmarks:
        hand_bboxes = get_hand_bboxes(
            results.multi_hand_landmarks, frame.shape[1], frame.shape[0]
        )

        for bbox, hand_landmarks in zip(hand_bboxes, results.multi_hand_landmarks):
            x_min, y_min, x_max, y_max = bbox
            cropped_hand = frame[y_min:y_max, x_min:x_max]

            hand_mask = None
            if cropped_hand.size > 0:
                hand_mask = create_hand_mask(
                    cropped_hand,
                    hand_landmarks,
                    bbox,
                    frame.shape[1],
                    frame.shape[0],
                )

            hand_outputs.append(
                {
                    "bbox": bbox,
                    "landmarks": hand_landmarks,
                    "cropped_hand": cropped_hand,
                    "mask": hand_mask,
                }
            )

    return hand_outputs


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

    # Smaller kernel for less dilation
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Less aggressive dilation to keep fingers thin
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # Lighter blur to smooth edges without making fingers fat
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Smaller closing kernel to preserve finger details
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def create_hand_mask(cropped_image, hand_landmarks, bbox, img_width, img_height):

    mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)

    x_min, y_min, x_max, y_max = bbox

    WRIST, THUMB, INDEX, MIDDLE, RING, PINKY = get_hand_number_landmarks()

    hand_points = convert_landmarks_to_cropped(
        hand_landmarks, cropped_image, img_width, img_height, bbox
    )

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    crop_size = min(crop_width, crop_height)

    proportion = crop_size / 200

    line_thickness = round(8 * proportion)
    joint_radius = line_thickness // 2

    connections = [
        (0, 1),
        (0, 5),
        (0, 17),  # Wrist to pinky base
        # Palm structure
        (5, 9),  # Index base to middle base
        (9, 13),  # Middle base to ring base
        (13, 17),  # Ring base to pinky base
        # Thumb
        (1, 2),
        (2, 3),
        (3, 4),
        # Index finger
        (5, 6),
        (6, 7),
        (7, 8),
        # Middle finger
        (9, 10),
        (10, 11),
        (11, 12),
        # Ring finger
        (13, 14),
        (14, 15),
        (15, 16),
        # Pinky finger
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    for start_idx, end_idx in connections:
        start_point = tuple(hand_points[start_idx])
        end_point = tuple(hand_points[end_idx])

        cv2.line(mask, start_point, end_point, 255, line_thickness)

    for point in hand_points:
        cv2.circle(mask, tuple(point), joint_radius, 255, -1)

    palm_points = [
        hand_points[0],
        hand_points[5],
        hand_points[9],
        hand_points[13],
        hand_points[17],
    ]

    palm_hull = cv2.convexHull(np.array(palm_points))
    cv2.fillPoly(mask, [palm_hull], 255)

    finger_thickness = line_thickness // 2

    for finger_indices in [THUMB, INDEX, MIDDLE, RING, PINKY]:

        finger_points = hand_points[finger_indices]

        for i in range(len(finger_points) - 1):

            thickness = finger_thickness

            if finger_indices == THUMB:

                thickness = finger_thickness + (10 * proportion)

            elif finger_indices == PINKY:

                thickness = max(1, finger_thickness - (4 * proportion))

            else:

                thickness = finger_thickness + (5 * proportion)

            cv2.line(
                mask,
                tuple(finger_points[i]),
                tuple(finger_points[i + 1]),
                255,
                int(thickness),
            )

        cv2.circle(mask, tuple(finger_points[-1]), finger_thickness // 2, 255, -1)

    triangle_pts = np.array(
        [
            (hand_points[5] + hand_points[6]) / 2,
            hand_points[0],
            (hand_points[1] + hand_points[2]) / 2,
        ],
        np.int32,
    )
    cv2.fillPoly(mask, [triangle_pts], 255)

    mask = apply_after_mask_effects(mask)

    return mask


def main():

    cap = cv2.VideoCapture(0)

    show_overlay = True

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        frame = cv2.flip(frame, 1)

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

                    if show_overlay:

                        hand_mask = create_hand_mask(
                            cropped_hand,
                            hand_landmarks,
                            bbox,
                            frame.shape[1],
                            frame.shape[0],
                        )

                        hand_mask_3ch = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
                        display_frame[y_min:y_max, x_min:x_max] = hand_mask_3ch

        else:

            display_frame = frame

        mode_text = "OVERLAY MODE" if show_overlay else "BOX ONLY MODE"
        cv2.putText(
            display_frame,
            mode_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            "Press 'o' to toggle overlay",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            "Magic Cam - Only Hand Recognition - press 'q' to quit", display_frame
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):  # Toggle overlay mode
            show_overlay = not show_overlay
            print(f"Switched to {'Overlay' if show_overlay else 'Box Only'} mode")

    hands.close()
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
