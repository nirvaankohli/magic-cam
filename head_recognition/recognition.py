import cv2
import mediapipe as mp

# Init MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Init Drawing Utils


def main():

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

    main()
