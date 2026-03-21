import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

VALID_GESTURES = ["open_hand", "fist", "point", "peace", "thumbs_up"]
SAMPLES_PER_GESTURE = 50
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_data.csv")
COUNTDOWN_SECONDS = 3

# CSV header: gesture label + 21 landmarks * 3 coords (x, y, z)
CSV_HEADER = ["gesture"] + [
    f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
]


def get_gesture_name():
    print("\n--- Gesture Collector ---")
    print("Available gestures:", ", ".join(VALID_GESTURES))
    while True:
        name = input("Enter gesture name to collect: ").strip().lower()
        if name in VALID_GESTURES:
            return name
        print(f"  Invalid gesture. Choose from: {', '.join(VALID_GESTURES)}")


def draw_overlay(frame, message, color=(255, 255, 255), y=50):
    cv2.putText(frame, message, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, message, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


def countdown(cap, hands, gesture_name, seconds=COUNTDOWN_SECONDS):
    """Show a live countdown so the user can pose before capture starts."""
    start = time.time()
    while True:
        remaining = seconds - int(time.time() - start)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        if remaining <= 0:
            draw_overlay(frame, f"Pose: {gesture_name}  GO!", color=(0, 255, 0), y=50)
            cv2.imshow("Gesture Collector", frame)
            cv2.waitKey(300)
            return

        draw_overlay(frame,
                     f"Pose: {gesture_name}  Starting in {remaining}...",
                     color=(0, 200, 255), y=50)
        cv2.imshow("Gesture Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


def collect_samples(gesture_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return []

    samples = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        countdown(cap, hands, gesture_name)

        print(f"\nCapturing {SAMPLES_PER_GESTURE} samples for '{gesture_name}'...")

        while len(samples) < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            captured_this_frame = False
            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract flat list of x, y, z for all 21 landmarks
                row = []
                for lm in hand_lm.landmark:
                    row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)])
                samples.append(row)
                captured_this_frame = True

            count = len(samples)
            progress = int((count / SAMPLES_PER_GESTURE) * 30)
            bar = "[" + "#" * progress + "." * (30 - progress) + "]"
            status = f"  {bar} {count}/{SAMPLES_PER_GESTURE}"

            color = (0, 255, 100) if captured_this_frame else (0, 100, 255)
            draw_overlay(frame, f"{gesture_name}  {count}/{SAMPLES_PER_GESTURE}",
                         color=color, y=50)

            if not captured_this_frame:
                draw_overlay(frame, "No hand detected — show your hand!",
                             color=(0, 80, 255), y=100)

            cv2.imshow("Gesture Collector", frame)
            print(status, end="\r")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nCollection aborted.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {len(samples)} samples for '{gesture_name}'.")
    return samples


def save_to_csv(gesture_name, samples):
    file_exists = os.path.isfile(CSV_FILE)
    mode = "a" if file_exists else "w"

    with open(CSV_FILE, mode=mode, newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        for row in samples:
            writer.writerow([gesture_name] + row)

    total = sum(1 for _ in open(CSV_FILE)) - 1  # subtract header
    print(f"Saved to: {CSV_FILE}")
    print(f"Total rows in dataset: {total}")


def main():
    gesture_name = get_gesture_name()
    samples = collect_samples(gesture_name)

    if samples:
        save_to_csv(gesture_name, samples)
        print(f"\nDone! {len(samples)} samples of '{gesture_name}' added to gesture_data.csv.")
    else:
        print("No samples collected. Nothing saved.")

    again = input("\nCollect another gesture? (y/n): ").strip().lower()
    if again == "y":
        main()


if __name__ == "__main__":
    main()
