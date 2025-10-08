import cv2 

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

def process_frame(frame, hands):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        
        for landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(

                image=frame,
                landmark_list=landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            
            )

    return frame

def main():

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(

        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():

            success, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not success:

                print("Ignoring empty camera frame.")

                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:

                for landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(

                        image=frame,

                        landmark_list=landmarks,
                        
                        connections=mp_hands.HAND_CONNECTIONS,
                        
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    
                    )

            cv2.imshow("Magic Cam - Only Hand Recognition - press 'q' to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break

    cap.release()

if __name__ == "__main__":

    main()

