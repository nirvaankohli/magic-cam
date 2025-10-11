# gesture_recognition_realtime.py
import cv2
import numpy as np
from background_subtraction import take_raw_frame_and_convert_to_contours
from cnn_inference import infer

CKPT_PATH = r"../gesture_cnn/outputs/best_V1_model.pth"
SAVE_DIR = r"C:/Users/nirva/OneDrive/Projects/magic-cam/gesture_recognition"

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

Class = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "High-Five",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "L",
    11: "Fist w/ index finger slightly up",
    12: "Crip",
    13: "Fingers Crossed",
    14: "Fist",
    15: "Thumbs Up Sideways",
    16: "Idk Even Know",
    17: "Drinking Sign",
    18: "Pinky Finger Up",
    19: "Thumbs Up",
}


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_num = 0

    print("Gesture Recognition with CNN Inference")
    print("Press 'q' to quit, 'f' to save frame")
    print("=" * 50)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hand_outputs = take_raw_frame_and_convert_to_contours(frame)

        if hand_outputs not in (None, []):
            mask = hand_outputs[0]["mask"]

            if mask is not None:
                if len(mask.shape) == 2:
                    disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                elif len(mask.shape) == 3:
                    disp = mask.copy()
                else:
                    disp = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

                input_img = mask
            else:
                disp = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                input_img = frame
        else:
            disp = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            input_img = frame

        # Ensure display image is always the correct size
        disp = cv2.resize(disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        try:
            results = infer(input_img, ckpt_path=CKPT_PATH, topk=3)

            if results:
                print(
                    f"Predictions: {[(label, f'{prob:.3f}') for label, prob in results]}"
                )
            else:
                print("No predictions returned")

            if results:
                # Display only top 3 predictions
                for i, (label, prob) in enumerate(results[:3]):
                    y_pos = 30 + i * 30
                    color = (
                        (0, 255, 0) if i == 0 else (255, 255, 255)
                    )  # Green for top prediction, white for others
                    font_scale = 0.7 if i == 0 else 0.5
                    thickness = 2 if i == 0 else 1

                    # Extract class number from label (assuming format "Class_X")
                    try:
                        if label.startswith("Class_"):
                            class_num = int(label.split("_")[1])
                            descriptive_name = Class.get(
                                class_num, f"Class_{class_num}"
                            )
                            display_text = (
                                f"{i+1}. {class_num} ({descriptive_name}): {prob:.3f}"
                            )
                        else:
                            display_text = f"{i+1}. {label}: {prob:.3f}"
                    except:
                        display_text = f"{i+1}. {label}: {prob:.3f}"

                    cv2.putText(
                        disp,
                        display_text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    disp,
                    "No prediction",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        except Exception as e:
            cv2.putText(
                disp,
                f"Error: {str(e)[:30]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            print(f"Inference error: {e}")

        cv2.imshow("Gesture Recognition with CNN", disp)

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
