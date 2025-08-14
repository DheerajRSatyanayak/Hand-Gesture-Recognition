import cv2
import mediapipe as mp

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for natural movement
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get handedness ("Left" or "Right")
                handedness = results.multi_handedness[idx].classification[0].label

                # Finger counting logic
                landmarks = hand_landmarks.landmark
                fingers = []
                tip_ids = [4, 8, 12, 16, 20]

                # Thumb detection changes based on hand type
                if handedness == "Right":
                    fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x else 0)
                else:  # Left hand
                    fingers.append(1 if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x else 0)

                # Other 4 fingers (same for both hands)
                for id in range(1, 5):
                    fingers.append(1 if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y else 0)

                # Count fingers
                count = fingers.count(1)

                # Display info
                cv2.putText(frame, f"{handedness} Hand", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {count}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show output
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

