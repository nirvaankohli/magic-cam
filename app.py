import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
from head_recognition import head_recognition
from head_recognition.import_test_w_hat import draw_hat


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.threshold1 = 100
        self.threshold2 = 200
        self.effect_options = ["None", "Grayscale", "Edge Detection", "Cartoon"]
        self.selected_effect = "None"
        self.effects = []
        self.bts = []
        self.display_texts = []
        # Initialize head recognizer once to avoid recreation overhead
        self.head_recognizer = head_recognition.HeadRecognition()

    def update_settings(self, effects, bts, display_texts):

        self.effects = effects
        self.bts = bts
        self.display_texts = display_texts

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        effects = self.effects
        bts = self.bts
        display_texts = self.display_texts

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        head_landmarks = False

        if "Wizard Hat" in effects:
            try:
                head_outputs = self.head_recognizer.process_frame(img)

                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if head_outputs["results"] is not None:

                    image_rgb_with_hat = draw_hat(image_rgb, head_outputs["results"])
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
            "Hand Subtraction",
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
