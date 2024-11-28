import cv2
import numpy as np
import mediapipe as mp

'''
Implementing MediaPipe for hand tracking with different colors for left and right hands.
'''

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("/home/niru/codes/disassembly/video_segmentation/data/input_video.mov")

# Define colors for left and right hands
left_hand_color = (0, 255, 0)   # Green for left hand
right_hand_color = (0, 0, 255)  # Red for right hand

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Determine the hand type (left or right)
                if handedness.classification[0].label == 'Left':
                    color = left_hand_color
                else:
                    color = right_hand_color

                # Draw landmarks with the specified color
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
