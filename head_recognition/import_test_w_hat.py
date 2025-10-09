import head_recognition
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import sys


def round_to_nearest(val, base=3):
    return int(base * round(float(val) / base))


def draw_hat(image_rgb, results):

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image_rgb.shape

        # landmarks for left, right, top of head

        left_idx = 67
        right_idx = 297
        top_idx = 10

        # config

        left = face_landmarks.landmark[left_idx]
        right = face_landmarks.landmark[right_idx]
        top = face_landmarks.landmark[top_idx]

        left_pt = np.array([int(left.x * w), int(left.y * h)])
        right_pt = np.array([int(right.x * w), int(right.y * h)])
        top_pt = np.array([int(top.x * w), int(top.y * h)])

        # angle of the hat

        delta = right_pt - left_pt
        angle = np.degrees(np.arctan2(delta[0], delta[1])) - 90

        angle = (angle // 2) * 2

        hat_img = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

        if hat_img is not None:

            print(angle)

            hat_size = 5

            hat_width = hat_size * int(np.linalg.norm(right_pt - left_pt))
            hat_height = int(1 * hat_width)
            resized_hat = cv2.resize(
                hat_img, (hat_width, hat_height), interpolation=cv2.INTER_AREA
            )

            anchor_point = (hat_width // 2, hat_height - 40)

            M = cv2.getRotationMatrix2D(anchor_point, angle, 1)
            rotated_hat = cv2.warpAffine(
                resized_hat,
                M,
                (hat_width, hat_height),
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            hat_x = int(top_pt[0] - anchor_point[0])
            hat_y = int(top_pt[1] - anchor_point[1])

            if image_rgb.shape[2] == 3:
                image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
            else:
                image_rgba = image_rgb.copy()

            y1 = hat_y
            y2 = hat_y + hat_height
            x1 = hat_x
            x2 = hat_x + hat_width

            pad_top = max(0, -y1)
            pad_left = max(0, -x1)
            pad_bottom = max(0, y2 - image_rgba.shape[0])
            pad_right = max(0, x2 - image_rgba.shape[1])

            if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                image_rgba = cv2.copyMakeBorder(
                    image_rgba,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0, 0],
                )

            y1 += pad_top
            y2 += pad_top
            x1 += pad_left
            x2 += pad_left

            overlay = np.zeros_like(image_rgba, dtype=np.uint8)
            overlay[y1:y2, x1:x2] = rotated_hat

            overlay_rgb = overlay[..., :3]
            overlay_alpha = overlay[..., 3] / 255.0
            overlay_alpha = np.expand_dims(overlay_alpha, axis=2)

            image_rgba[..., :3] = (
                overlay_rgb * overlay_alpha + image_rgba[..., :3] * (1 - overlay_alpha)
            ).astype(np.uint8)

            if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                h, w = image_rgb.shape[:2]
                image_rgba = image_rgba[pad_top : pad_top + h, pad_left : pad_left + w]

            image_rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB)

            return image_rgb


def main():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        head_recognizer = head_recognition.HeadRecognition()
        head_outputs = head_recognizer.process_frame(frame)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if head_outputs["results"] is not None:

            image_rgb = draw_hat(image_rgb, head_outputs["results"])

        cv2.imshow("MediaPipe Face", image_rgb)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()


if __name__ == "__main__":

    main()
