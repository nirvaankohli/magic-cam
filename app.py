import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import os
from head_recognition import head_recognition
from head_recognition.import_test_w_hat import draw_hat
from pretrained_gesture_recognition import recognition
from hand_recognition import recognitionv2 as rec


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.threshold1 = 100
        self.threshold2 = 200
        self.effect_options = ["None", "Grayscale", "Edge Detection", "Cartoon"]
        self.selected_effect = "None"
        self.effects = []
        self.bts = []
        self.display_texts = []
        self.current_effect = "None"
        self.current_effect_stage = {
            "current": 0,
            "max": 0,
            "stage": 0,
            "bottom_center": None,
        }
        self.frames_before = 3
        self.head_recognizer = head_recognition.HeadRecognition()
        self.overlay_img = None

    def update_settings(self, effects, bts, display_texts):

        self.effects = effects
        self.bts = bts
        self.display_texts = display_texts

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        try:
            effects = self.effects
            bts = self.bts
            display_texts = self.display_texts

            img = frame.to_ndarray(format="bgr24")

            img = cv2.flip(img, 1)

            head_landmarks = False

            if "Spells" in effects:

                hand_recognition_result = recognition.recognize_gesture(img)

                if "Model Output(hand)" in display_texts:

                    if hand_recognition_result.gestures:

                        img = recognition.draw_results(hand_recognition_result, img)

                    else:
                        cv2.putText(
                            img,
                            "No hand detected",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                top_gesture = None
                gesture_name = None
                confidence = None

                processed_frame, self.hand_results = rec.process_frame(img)

                if "Hand Landmarks" in bts and self.hand_results is not None:
                    img, self.hand_results = rec.process_frame(img, draw_results=True)
                else:
                    processed_frame, self.hand_results = rec.process_frame(
                        img, draw_results=False
                    )

                if hand_recognition_result.gestures:
                    print("Hand gestures detected")
                    top_gesture = hand_recognition_result.gestures[0][0]
                    gesture_name = top_gesture.category_name
                    confidence = top_gesture.score
                    print(f"Gesture: {gesture_name}, Confidence: {confidence}")

                    for i, gesture_list in enumerate(hand_recognition_result.gestures):
                        for j, gesture in enumerate(gesture_list):
                            print(
                                f"  Gesture {i}-{j}: {gesture.category_name} (confidence: {gesture.score:.3f})"
                            )

                    if gesture_name in ["Pointing_Up"] and confidence > 0.4:
                        print(
                            f"Fireball trigger gesture detected: {gesture_name} with confidence {confidence:.3f}"
                        )

                        if self.current_effect == "None":
                            print("Starting new fireball effect")
                            self.current_effect = "Fireball"
                            self.current_effect_stage = {
                                "stage": 0,
                                "current": 0,
                                "max": 15,
                                "bottom_center": None,
                            }

                        if (
                            self.current_effect == "Fireball"
                            and self.current_effect_stage["stage"] <= self.frames_before
                        ):
                            self.current_effect_stage["current"] += 1
                            print(
                                f"Fireball charging stage: {self.current_effect_stage['current']}/{self.frames_before}"
                            )

                            if (
                                self.current_effect_stage["current"]
                                >= self.frames_before
                            ):
                                print(
                                    "Charging complete, advancing to fireball animation"
                                )
                                self.current_effect_stage["stage"] = (
                                    self.frames_before + 1
                                )

                else:
                    print("No hand gestures detected")

                if (
                    self.current_effect == "Fireball"
                    and self.current_effect_stage["stage"] >= self.frames_before + 1
                ):
                    try:
                        print(
                            f"Fireball effect active - Stage: {self.current_effect_stage['stage']}, Frame: {self.current_effect_stage['current']}"
                        )

                        if (
                            self.current_effect_stage["stage"] == self.frames_before + 1
                            and self.hand_results is not None
                            and self.hand_results.multi_hand_landmarks
                        ):
                            self.current_effect_stage["stage"] = self.frames_before + 2

                            hand_landmarks = self.hand_results.multi_hand_landmarks[0]
                            bbox = rec.get_hand_bbox(hand_landmarks, img)
                            print(f"Hand bounding box: {bbox}")

                            self.current_effect_stage["bottom_center"] = (
                                (bbox[0] + bbox[2]) // 2,
                                bbox[1],
                            )

                            print(
                                f"Fireball position set to: {self.current_effect_stage['bottom_center']}"
                            )

                            self.current_effect_stage["current"] = (
                                self.frames_before + 1
                            )

                        if self.current_effect_stage["stage"] == self.frames_before + 2:
                            self.current_effect_stage["current"] += 1
                            print(
                                f"Animating fireball frame: {self.current_effect_stage['current']}"
                            )

                            # Load the fireball frame
                            if self.current_effect_stage["current"] <= 15:
                                g = str(self.current_effect_stage["current"])
                                if self.current_effect_stage["current"] <= 9:
                                    g = "0" + g

                                # Use PNG files instead of GIF
                                fireball_path = os.path.join(
                                    "assets", "fireball_png", f"frame_{g}.png"
                                )
                                print(f"Loading fireball image: {fireball_path}")
                                try:
                                    self.overlay_img = cv2.imread(
                                        fireball_path, cv2.IMREAD_UNCHANGED
                                    )
                                    if self.overlay_img is not None:
                                        print(
                                            f"Successfully loaded fireball image with shape: {self.overlay_img.shape}"
                                        )
                                    else:
                                        print(
                                            f"Failed to load fireball image: {fireball_path}"
                                        )
                                except Exception as e:
                                    print(f"Error loading fireball image: {e}")
                            else:
                                print("Fireball animation complete, resetting effect")
                                self.current_effect = "None"
                                self.current_effect_stage = {
                                    "current": 0,
                                    "max": 0,
                                    "stage": 0,
                                    "bottom_center": None,
                                }
                                self.overlay_img = None

                            if (
                                self.current_effect_stage.get("bottom_center")
                                is not None
                                and self.overlay_img is not None
                            ):
                                print("Overlaying fireball on image")

                                scale_factor = 0.3
                                overlay_h, overlay_w = self.overlay_img.shape[:2]
                                new_h = int(overlay_h * scale_factor)
                                new_w = int(overlay_w * scale_factor)

                                resized_overlay = cv2.resize(
                                    self.overlay_img, (new_w, new_h)
                                )

                                x_center = self.current_effect_stage["bottom_center"][0]
                                y_bottom = self.current_effect_stage["bottom_center"][1]

                                x_start = x_center - new_w // 2
                                y_start = y_bottom - new_h

                                img_h, img_w = img.shape[:2]
                                print(
                                    f"Overlay position: ({x_start}, {y_start}) to ({x_start + new_w}, {y_start + new_h})"
                                )
                                print(f"Image bounds: {img_w} x {img_h}")

                                if (
                                    x_start >= 0
                                    and y_start >= 0
                                    and x_start + new_w <= img_w
                                    and y_start + new_h <= img_h
                                ):
                                    print("Overlay within bounds, applying to image")
                                    x_end = x_start + new_w
                                    y_end = y_start + new_h

                                    if resized_overlay.shape[2] == 4:
                                        print(
                                            "Applying RGBA overlay with alpha blending"
                                        )
                                        alpha = resized_overlay[:, :, 3] / 255.0
                                        for c in range(3):
                                            img[y_start:y_end, x_start:x_end, c] = (
                                                alpha * resized_overlay[:, :, c]
                                                + (1 - alpha)
                                                * img[y_start:y_end, x_start:x_end, c]
                                            )
                                    else:
                                        print("Applying RGB overlay")
                                        img[y_start:y_end, x_start:x_end] = (
                                            resized_overlay[:, :, :3]
                                        )
                                else:
                                    print("Overlay position out of bounds, skipping")
                            else:
                                if (
                                    self.current_effect_stage.get("bottom_center")
                                    is None
                                ):
                                    print("No bottom_center position available")
                                if self.overlay_img is None:
                                    print("No overlay image available")

                    except Exception as e:
                        print(f"Error in fireball effect: {e}")
                        import traceback

                        traceback.print_exc()

            if "Wizard Hat" in effects:
                try:
                    head_outputs = self.head_recognizer.process_frame(img)

                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if head_outputs["results"] is not None:

                        image_rgb_with_hat = draw_hat(
                            image_rgb, head_outputs["results"]
                        )
                        if image_rgb_with_hat is not None:
                            image_rgb = image_rgb_with_hat

                    if "Head Landmarks" in bts:

                        head_recognition.draw_image(image_rgb, head_outputs["results"])
                        head_landmarks = True

                    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:

                    print(f"Head recognition error: {e}")

            if "Head Landmarks" in bts and not head_landmarks:
                try:
                    head_outputs = self.head_recognizer.process_frame(img)

                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if head_outputs["results"] is not None:

                        head_recognition.draw_image(image_rgb, head_outputs["results"])

                    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:

                    print(f"Head landmarks error: {e}")

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:

            print(f"Error in recv: {e}")
            import traceback

            traceback.print_exc()
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")


VideoProcessor()
st.title("Webcam with OpenCV Processing")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


with st.sidebar:

    st.header("Settings")

    st.subheader("Effects")

    effects = st.multiselect(
        "Select effects to apply",
        [
            "Wizard Hat",
            "Spells",
        ],
        default=[
            "Wizard Hat",
            "Spells",
        ],
    )

    st.subheader("Behind the Scenes")

    bts = st.multiselect(
        "Select behind the scenes effects to apply",
        [
            "Hand Landmarks",
            "Head Landmarks",
        ],
    )

    st.subheader("Displayed Text")

    display_texts = st.multiselect(
        "Select displayed text to apply",
        [
            "Model Output(hand)",
        ],
    )

st.write(display_texts, effects, bts)

ctx = webrtc_streamer(
    key="opencv-filter",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.update_settings(effects, bts, display_texts)
