import cv2
import mediapipe as mp
import time

class landmarker_and_result:

    def __init__(self):

        self.result = None
        self.landmarker = mp.tasks.vision.HandLandmarker
        
        self.create_land_marker()

    def create_land_marker(self):
        
        def update_result(result, output_image, timestamp_ms):
            
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions( 

            base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands = 2, 
            min_hand_detection_confidence = 0.3, 
            min_hand_presence_confidence = 0.3, 
            min_tracking_confidence = 0.3, 
            result_callback=update_result
        
            )
        
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            self.landmarker.detect_async(mp_image, int(time.time() * 1000))
        except:
            pass

    def close(self):

        self.landmarker.close() 

def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result:
        return rgb_image
    hand_landmarks_list = getattr(detection_result, 'hand_landmarks', None)
    if not hand_landmarks_list:
        return rgb_image
    h, w, _ = rgb_image.shape
    for hand_landmarks in hand_landmarks_list:
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(rgb_image, (cx, cy), 4, (0, 255, 0), -1)
    return rgb_image
    


def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    hand_landmarker = landmarker_and_result()

    while True:
        # Capture frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        hand_landmarker.detect_async(frame)
        frame = draw_landmarks_on_image(frame, hand_landmarker.result)

        # Show the frame
        cv2.imshow('frame', frame)

        # if 'q' is pressed, quit
        if cv2.waitKey(1) == ord('q'):
            break

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   
    main()