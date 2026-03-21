import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",  "THUMB_MCP",  "THUMB_IP",   "THUMB_TIP",
    "INDEX_MCP",  "INDEX_PIP",  "INDEX_DIP",  "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP",   "RING_PIP",   "RING_DIP",   "RING_TIP",
    "PINKY_MCP",  "PINKY_PIP",  "PINKY_DIP",  "PINKY_TIP",
]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        print("Hand tracker started. Press 'q' to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    print(f"--- Hand {hand_idx + 1} ({label}) ---")

                    for i, lm in enumerate(hand_landmarks.landmark):
                        print(
                            f"  [{i:02d}] {LANDMARK_NAMES[i]:<14} "
                            f"x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}"
                        )
                    print()

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
