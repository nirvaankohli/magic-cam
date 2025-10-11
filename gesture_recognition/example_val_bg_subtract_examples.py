import cv2
import mediapipe as mp
from background_subtraction import take_raw_frame_and_convert_to_contours
import numpy as np


def main():

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_num = 0

    print("Background Subtraction Validation")
    print("Press 'q' to quit, 'f' to save frame")
    print("=" * 40)

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        hand_outputs = take_raw_frame_and_convert_to_contours(frame)

        if hand_outputs not in [None, []]:
            mask = hand_outputs[0]["mask"]

            if mask is not None:
                # Display the mask (already properly formatted)
                if len(mask.shape) == 3:
                    display_mask = mask
                else:
                    display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Hand Mask", display_mask)

                # Also show bbox info
                bbox = hand_outputs[0]["bbox"]
                print(f"Hand detected - bbox: {bbox}, mask shape: {mask.shape}")
            else:
                cv2.imshow("Hand Mask", np.zeros((240, 320, 3), dtype=np.uint8))
        else:
            cv2.imshow("Hand Mask", np.zeros((240, 320, 3), dtype=np.uint8))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("f"):
            if hand_outputs not in [None, []]:
                mask = hand_outputs[0]["mask"]
                if mask is not None:
                    filename = f"C:/Users/nirva/OneDrive/Projects/magic-cam/gesture_recognition/outputs/frame{frame_num}.png"
                    cv2.imwrite(filename, mask)
                    print(f"Saved {filename}")
                    frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
