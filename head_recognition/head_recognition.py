import cv2
import mediapipe as mp
import numpy as np


class HeadRecognition:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame):

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(image_rgb)

        return_arr = {"results": None, "bbox": None}

        if results.multi_face_landmarks:
            return_arr["results"] = results
            return_arr["bbox"] = get_bbox(results, image_rgb)

        return return_arr


# if not for import use
def setup_drawing_utils():

    return mp.solutions.drawing_utils, mp.solutions.drawing_styles


def get_bbox(results, image_rgb):

    h, w, _ = image_rgb.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

    return (x_min, y_min, x_max, y_max)


# draw the face thingies on the image
def draw_image(image_rgb, results):
    mp_drawing, mp_drawing_styles = setup_drawing_utils()

    for face_landmarks in results.multi_face_landmarks:

        mp_drawing.draw_landmarks(
            image=image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        mp_drawing.draw_landmarks(
            image=image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

        mp_drawing.draw_landmarks(
            image=image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        return image_rgb


def draw_bbox(image_rgb, bbox):

    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


def draw_example_hat(image_rgb, bbox, results):

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

        # load the hat image with alpha channel

        hat_img = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

        if hat_img is not None:

            print(angle)

            # resize the hat to fit the width of the face

            hat_size = 5

            hat_width = hat_size * int(np.linalg.norm(right_pt - left_pt))
            hat_height = int(0.6 * hat_width)
            resized_hat = cv2.resize(
                hat_img, (hat_width, hat_height), interpolation=cv2.INTER_AREA
            )

            M = cv2.getRotationMatrix2D((hat_width // 2, hat_height // 2), angle, 1)

            rotated_hat = cv2.warpAffine(
                resized_hat,
                M,
                (hat_width, hat_height),
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            center_pt = ((left_pt + right_pt) // 4) * 2

            hat_x = int(center_pt[0] - hat_width // 2)
            hat_y = int(top_pt[1] - hat_height * 0.8)

            y1, y2 = max(0, hat_y), min(image_rgb.shape[0], hat_y + hat_height)
            x1, x2 = max(0, hat_x), min(image_rgb.shape[1], hat_x + hat_width)

            hat_y1 = max(0, -hat_y)
            hat_y2 = hat_height - max(0, (hat_y + hat_height) - image_rgb.shape[0])
            hat_x1 = max(0, -hat_x)
            hat_x2 = hat_width - max(0, (hat_x + hat_width) - image_rgb.shape[1])

            if y2 > y1 and x2 > x1:

                hat_rgba = rotated_hat[hat_y1:hat_y2, hat_x1:hat_x2]
                hat_rgb = hat_rgba[..., :3]
                hat_alpha = hat_rgba[..., 3:] / 255.0

                roi = image_rgb[y1:y2, x1:x2]

                image_rgb[y1:y2, x1:x2] = (
                    hat_rgb * hat_alpha + roi * (1 - hat_alpha)
                ).astype(np.uint8)

    return image_rgb


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

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get Results
        results = face_mesh.process(image_rgb)

        # draw the face mesh annotations on the image.
        if results.multi_face_landmarks:

            # print(f"BOX: {get_bbox(results, image_rgb)}")
            # draw_image(image_rgb, results)
            # draw_bbox(image_rgb, get_bbox(results, image_rgb))
            draw_example_hat(image_rgb, get_bbox(results, image_rgb), results)

        # Show the image
        cv2.imshow("The frame", image_rgb)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    mp_drawing, mp_drawing_styles = setup_drawing_utils()

    main(mp_drawing, mp_drawing_styles)
