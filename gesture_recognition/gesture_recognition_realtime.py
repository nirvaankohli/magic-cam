# gesture_recognition_realtime.py
import cv2
import numpy as np
import os
from background_subtraction import take_raw_frame_and_convert_to_contours
from cnn_inference import infer

# Use absolute path to avoid path resolution issues
CKPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "gesture_cnn", "outputs", "best_V1_model.pth"
)
SAVE_DIR = r"C:/Users/nirva/OneDrive/Projects/magic-cam/gesture_recognition"

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
SQUARE_SIZE = 300  # Size for square cropping before feeding to model

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


def make_square_image(image, size=SQUARE_SIZE):
    """
    Convert an image to a square by padding or cropping, then resize to specified size.
    """
    if image is None:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Handle grayscale images
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]

    # Make it square by padding the smaller dimension
    if h == w:
        # Already square
        square_img = image
    elif h > w:
        # Pad width
        pad_width = (h - w) // 2
        pad_left = pad_width
        pad_right = h - w - pad_left
        square_img = cv2.copyMakeBorder(
            image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        # Pad height
        pad_height = (w - h) // 2
        pad_top = pad_height
        pad_bottom = w - h - pad_top
        square_img = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    # Resize to target size
    square_img = cv2.resize(square_img, (size, size))
    return square_img


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
    print(f"Model path: {CKPT_PATH}")
    print("=" * 50)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # Get hand detection results
        hand_outputs = take_raw_frame_and_convert_to_contours(frame)

        # Prepare input for CNN
        input_img = None
        display_img = None

        if hand_outputs and len(hand_outputs) > 0:
            # Use the first detected hand
            hand_data = hand_outputs[0]
            mask = hand_data.get("mask")

            if mask is not None and mask.size > 0:
                # Convert mask to square format for CNN input
                input_img = make_square_image(mask, SQUARE_SIZE)

                # Create display version (also square but might be different from input)
                display_img = make_square_image(
                    mask, min(DISPLAY_WIDTH, DISPLAY_HEIGHT)
                )
            else:
                # Fallback to cropped hand if mask is not available
                cropped_hand = hand_data.get("cropped_hand")
                if cropped_hand is not None and cropped_hand.size > 0:
                    input_img = make_square_image(cropped_hand, SQUARE_SIZE)
                    display_img = make_square_image(
                        cropped_hand, min(DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    )

        # If no hand detected, use full frame
        if input_img is None:
            input_img = make_square_image(frame, SQUARE_SIZE)
            display_img = make_square_image(frame, min(DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Ensure display image is correct size for the window
        if display_img is not None:
            disp = cv2.resize(display_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        else:
            disp = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Run CNN inference
        try:
            results = infer(input_img, ckpt_path=CKPT_PATH, topk=3)

            if results:
                print(
                    f"Predictions: {[(label, f'{prob:.3f}') for label, prob in results]}"
                )
            else:
                print("No predictions returned")

            # Display predictions on image
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
            error_msg = f"Error: {str(e)[:50]}"
            cv2.putText(
                disp,
                error_msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            print(f"Inference error: {e}")

        # Add status information to display
        status_y = DISPLAY_HEIGHT - 60
        cv2.putText(
            disp,
            f"Input size: {SQUARE_SIZE}x{SQUARE_SIZE}",
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )
        cv2.putText(
            disp,
            f"Hands detected: {'Yes' if hand_outputs else 'No'}",
            (10, status_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

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
