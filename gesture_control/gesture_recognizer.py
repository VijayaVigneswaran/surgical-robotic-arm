import cv2
import pickle
import numpy as np
import os
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE   = os.path.join(DIR, "gesture_model.pkl")
ENCODER_FILE = os.path.join(DIR, "label_encoder.pkl")

# ── Gesture → Robot command mapping ───────────────────────────────────────────
GESTURE_COMMANDS = {
    "open_hand":  ("HOME",           (0,   200, 100)),   # green-teal
    "fist":       ("STOP / CLOSE",   (0,   60,  220)),   # red-ish (BGR)
    "point":      ("MOVE FORWARD",   (255, 180, 0  )),   # blue
    "peace":      ("MOVE UP",        (255, 100, 200)),   # purple
    "thumbs_up":  ("CONFIRM/EXECUTE",(0,   220, 255)),   # yellow
}

# Smoothing: majority vote over last N predictions
SMOOTH_WINDOW = 7

# Confidence threshold — predictions below this show as "uncertain"
CONF_THRESHOLD = 0.45


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_text_with_bg(frame, text, pos, font_scale, color, thickness=2,
                       bg_color=(20, 20, 20), pad=10):
    """Draw text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bar(frame, confidence, x, y, width=220, height=18):
    """Draw a horizontal confidence bar."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), cv2.FILLED)
    fill = int(width * confidence)
    bar_color = (0, 200, 80) if confidence >= CONF_THRESHOLD else (0, 80, 200)
    cv2.rectangle(frame, (x, y), (x + fill, y + height), bar_color, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (180, 180, 180), 1)
    label = f"{confidence * 100:.0f}%"
    cv2.putText(frame, label, (x + width + 8, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame, gesture, command, cmd_color, confidence, history):
    h, w = frame.shape[:2]

    # ── Top banner ────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 115), (15, 15, 15), cv2.FILLED)

    if gesture:
        # Gesture name (large)
        draw_text_with_bg(frame, gesture.replace("_", " ").upper(),
                          (18, 52), 1.35, cmd_color, thickness=2,
                          bg_color=(15, 15, 15), pad=4)

        # Robot command (medium, white)
        draw_text_with_bg(frame, f"CMD:  {command}",
                          (18, 98), 0.78, (230, 230, 230), thickness=1,
                          bg_color=(15, 15, 15), pad=4)

        # Confidence bar
        draw_confidence_bar(frame, confidence, w - 260, 20)
        cv2.putText(frame, "confidence", (w - 260, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    else:
        draw_text_with_bg(frame, "No hand detected",
                          (18, 60), 1.0, (100, 100, 255), thickness=2,
                          bg_color=(15, 15, 15), pad=4)

    # ── Smoothing history dots ─────────────────────────────────────────────────
    dot_x = w - 260
    dot_y = 60
    cv2.putText(frame, "buffer", (dot_x, dot_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
    for i, g in enumerate(history):
        color = GESTURE_COMMANDS.get(g, ("", (120, 120, 120)))[1] if g else (60, 60, 60)
        cv2.circle(frame, (dot_x + i * 22, dot_y + 12), 8, color, cv2.FILLED)
        cv2.circle(frame, (dot_x + i * 22, dot_y + 12), 8, (200, 200, 200), 1)

    # ── Legend (bottom strip) ─────────────────────────────────────────────────
    legend_y = h - 10
    items = list(GESTURE_COMMANDS.items())
    col_w = w // len(items)
    for i, (gest, (cmd, col)) in enumerate(items):
        lx = i * col_w + 10
        cv2.putText(frame, gest.replace("_", " "), (lx, legend_y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"→ {cmd}", (lx, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Thin separator lines ───────────────────────────────────────────────────
    cv2.line(frame, (0, 115), (w, 115), (60, 60, 60), 1)
    cv2.line(frame, (0, h - 30), (w, h - 30), (60, 60, 60), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        le = pickle.load(f)
    print("Model and encoder loaded.")
    print(f"  Classes: {list(le.classes_)}\n")
    return clf, le


def main():
    clf, le = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prediction_history = []   # rolling window for smoothing
    last_command = None
    last_logged  = None
    fps_time     = time.time()
    fps          = 0

    print("Gesture Recognizer running. Press 'q' to quit.\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # FPS counter
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 1e-5))
            fps_time = now

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture    = None
            command    = ""
            cmd_color  = (200, 200, 200)
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Build feature vector
                features = []
                for lm in hand_lm.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                features = np.array(features, dtype=np.float32).reshape(1, -1)

                # Predict
                proba       = clf.predict_proba(features)[0]
                pred_idx    = np.argmax(proba)
                confidence  = proba[pred_idx]
                raw_gesture = le.inverse_transform([pred_idx])[0]

                # Smooth over window
                prediction_history.append(raw_gesture)
                if len(prediction_history) > SMOOTH_WINDOW:
                    prediction_history.pop(0)

                # Majority vote
                if len(prediction_history) == SMOOTH_WINDOW:
                    from collections import Counter
                    gesture = Counter(prediction_history).most_common(1)[0][0]
                else:
                    gesture = raw_gesture

                # Only accept if confidence is high enough
                if confidence < CONF_THRESHOLD:
                    gesture = None

                if gesture:
                    command, cmd_color = GESTURE_COMMANDS.get(
                        gesture, ("UNKNOWN", (180, 180, 180))
                    )
                    # Log new commands to terminal
                    if command != last_logged:
                        print(f"  [{gesture:<14}]  →  {command}")
                        last_logged = command

            else:
                prediction_history.clear()

            # Draw HUD
            draw_hud(frame, gesture, command, cmd_color, confidence,
                     prediction_history + [""] * (SMOOTH_WINDOW - len(prediction_history)))

            # FPS badge
            cv2.putText(frame, f"FPS {fps:.0f}", (w - 80, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

            cv2.imshow("Surgical Arm — Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nRecognizer stopped.")


if __name__ == "__main__":
    main()
