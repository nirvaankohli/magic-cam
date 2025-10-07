import cv2
import mediapipe as mp
import time

class landmarker_and_result:

    def __init__(self):

        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.create_land_marker()

    def create_land_marker(self):
        
        def update_result(result, output_image, timestamp_ms):
            
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions( 

            base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
            num_hands = 2, 
            min_hand_detection_confidence = 0.3, 
            min_hand_presence_confidence = 0.3, 
            min_tracking_confidence = 0.3, 
            result_callback=update_result
        
            )
        
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def close(self):

        self.landmarker.close() 
        
def main():

    cap = cv2.VideoCapture(0)

    while True:

        # Capture frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Show the frame
        cv2.imshow('frame', frame)

        # if 'q' is pressed, quit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   
    main()