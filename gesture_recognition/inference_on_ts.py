# webcam_infer_simple.py
import cv2
import numpy as np
from background_subtraction import take_raw_frame_and_convert_to_contours
from cnn_inference import infer

CKPT_PATH = r"./outputs/best_V1_model.pth"
SAVE_DIR = r"C:/Users/nirva/OneDrive/Projects/magic-cam/gesture_recognition"


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")
    frame_num = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hand_outputs = take_raw_frame_and_convert_to_contours(frame)

        if hand_outputs not in (None, []):
            mask = hand_outputs[0]["mask"]
            if len(mask.shape) == 2 or mask.shape[2] == 1:
                disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                disp = mask.copy()
            input_img = hand_outputs[0]["mask"]
        else:
            disp = np.zeros((240, 320, 3), dtype=np.uint8)
            input_img = frame

        try:
            top1_label, top1_prob = infer(input_img, ckpt_path=CKPT_PATH, topk=1)[0]
            cv2.putText(
                disp,
                f"{top1_label} ({top1_prob:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception as e:
            cv2.putText(
                disp,
                f"Infer err: {e}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            print(f"Inference error: {e}")

        cv2.imshow("MediaPipe Hands", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            path = f"{SAVE_DIR}/frame{frame_num}.png"
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
