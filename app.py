import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from head_recognition import head_recognition
from head_recognition.import_test_w_hat import draw_hat
from pretrained_gesture_recognition import recognition
from hand_recognition import recognitionv2 as rec


if "current_effect" not in st.session_state:
    st.session_state.current_effect = "None"
    st.session_state.current_effect_stage = {
        "current": 0,
        "max": 0,
        "stage": 0,
        "bottom_center": None,
    }
    st.session_state.frames_before = 3
    st.session_state.overlay_img = None


@st.cache_resource
def get_head_recognizer():
    return head_recognition.HeadRecognition()


def process_image(img, effects, bts, display_texts):

    try:

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

            processed_frame, hand_results = rec.process_frame(img)

            if "Hand Landmarks" in bts and hand_results is not None:
                img, hand_results = rec.process_frame(img, draw_results=True)
            else:
                processed_frame, hand_results = rec.process_frame(
                    img, draw_results=False
                )

            if hand_recognition_result.gestures:
                top_gesture = hand_recognition_result.gestures[0][0]
                gesture_name = top_gesture.category_name
                confidence = top_gesture.score

                if gesture_name in ["Pointing_Up"] and confidence > 0.4:

                    if hand_results is not None and hand_results.multi_hand_landmarks:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        bbox = rec.get_hand_bbox(hand_landmarks, img)

                        bottom_center = ((bbox[0] + bbox[2]) // 2, bbox[1])

                        fireball_path = os.path.join(
                            "assets", "fireball_png", "frame_10.png"
                        )
                        try:
                            overlay_img = cv2.imread(
                                fireball_path, cv2.IMREAD_UNCHANGED
                            )
                            if overlay_img is not None:
                                scale_factor = 0.3
                                overlay_h, overlay_w = overlay_img.shape[:2]
                                new_h = int(overlay_h * scale_factor)
                                new_w = int(overlay_w * scale_factor)

                                resized_overlay = cv2.resize(
                                    overlay_img, (new_w, new_h)
                                )

                                x_center = bottom_center[0]
                                y_bottom = bottom_center[1]
                                x_start = x_center - new_w // 2
                                y_start = y_bottom - new_h

                                img_h, img_w = img.shape[:2]

                                if (
                                    x_start >= 0
                                    and y_start >= 0
                                    and x_start + new_w <= img_w
                                    and y_start + new_h <= img_h
                                ):

                                    x_end = x_start + new_w
                                    y_end = y_start + new_h

                                    if resized_overlay.shape[2] == 4:
                                        alpha = resized_overlay[:, :, 3] / 255.0
                                        for c in range(3):
                                            img[y_start:y_end, x_start:x_end, c] = (
                                                alpha * resized_overlay[:, :, c]
                                                + (1 - alpha)
                                                * img[y_start:y_end, x_start:x_end, c]
                                            )
                                    else:
                                        img[y_start:y_end, x_start:x_end] = (
                                            resized_overlay[:, :, :3]
                                        )
                        except Exception as e:
                            st.error(f"Error loading fireball effect: {e}")

        if "Wizard Hat" in effects:
            try:
                head_recognizer = get_head_recognizer()
                head_outputs = head_recognizer.process_frame(img)

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
                st.error(f"Head recognition error: {e}")

        if "Head Landmarks" in bts and not head_landmarks:
            try:
                head_recognizer = get_head_recognizer()
                head_outputs = head_recognizer.process_frame(img)

                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if head_outputs["results"] is not None:
                    head_recognition.draw_image(image_rgb, head_outputs["results"])

                img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Head landmarks error: {e}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return img


st.title("Magic Cam - Photo Effects")

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

st.write(
    "Selected options:",
    {"Effects": effects, "Behind the Scenes": bts, "Display Text": display_texts},
)

picture = st.camera_input("Take a picture")

if picture is not None:

    img = Image.open(picture)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("With Effects")
        with st.spinner("Processing image..."):
            processed_img = process_image(img, effects, bts, display_texts)
            st.image(processed_img, use_container_width=True)

    if processed_img:

        import io

        buf = io.BytesIO()
        processed_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download processed image",
            data=byte_im,
            file_name="magic_cam_processed.png",
            mime="image/png",
        )

st.info("💡 Tip: Try pointing up with your finger to cast a fireball spell!")
st.info("🎭 The wizard hat will automatically appear on detected faces!")
