import cv2
import mediapipe as mp
import numpy as np


# if not for import use
def setup_drawing_utils():

    return mp.solutions.drawing_utils, mp.solutions.drawing_styles


# Init MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def main(mp_drawing, mp_drawing_styles):

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        frame = cv2.flip(frame, 1)

        cv2.imshow("The frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    mp_drawing, mp_drawing_styles = setup_drawing_utils()

    main(mp_drawing, mp_drawing_styles)
