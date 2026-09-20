"""
Real-time webcam gesture recognizer.

Detects: Thumbs Up, Peace Sign, Korean Finger Heart, OK Sign,
Prayer Hands (two hands), and Index Raised (with an image-based glasses
overlay). Each active gesture shows a styled label, a large emoji badge in
the top-right corner, and a hand-drawn illustration in the left-side panel.

Controls:
  q - quit
  d - toggle debug overlay (per-finger curl states)
  m - toggle mirror mode
"""

import sys
import time
import platform
from collections import deque, defaultdict

if sys.version_info[:2] >= (3, 13):
    sys.exit(
        "MediaPipe does not currently publish Python 3.13 wheels, so "
        f"'import mediapipe' will fail on this interpreter ({sys.version.split()[0]}).\n"
        "Run this app in a Python 3.11 or 3.12 virtual environment instead, e.g.:\n"
        "  uv venv --python 3.11 venv && source venv/bin/activate && pip install -r requirements.txt\n"
        "See the README's 'Python 3.13' section for details."
    )

import cv2
import mediapipe as mp
import numpy as np

from gesture_recognizer import analyze_hand, classify_frame, GESTURE_EMOJI
from overlay import (
    ACCENT_SINGLE,
    ACCENT_TWO_HAND,
    build_icon_cache,
    build_drawing_cache,
    draw_drawing_panel,
    draw_gesture_icons,
    draw_glasses_image,
    draw_hint_bar,
    draw_label,
    load_glasses_image,
)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

SMOOTHING_WINDOW = 6        # frames of history per hand
MIN_STABLE_FRACTION = 0.6   # fraction of window that must agree before displaying

GLASSES_GESTURE = "Index Raised"


class GestureStabilizer:
    """Keeps a short rolling history per hand label to reduce flicker."""

    def __init__(self, window=SMOOTHING_WINDOW, min_fraction=MIN_STABLE_FRACTION):
        self.window = window
        self.min_fraction = min_fraction
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, key, label):
        self.history[key].append(label)
        hist = self.history[key]
        counts = {}
        for v in hist:
            counts[v] = counts.get(v, 0) + 1
        best_label, best_count = max(counts.items(), key=lambda kv: kv[1])
        if best_label is not None and best_count / len(hist) >= self.min_fraction:
            return best_label
        return None


def draw_debug(frame, hand_info, origin):
    x, y = origin
    cv2.putText(frame, hand_info.handedness, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (finger, curl) in enumerate(hand_info.curls.items()):
        cv2.putText(
            frame, f"{finger}: {curl.value}", (x, y + 20 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA,
        )


def main():
    # Detect operating system for camera backend
    is_mac = platform.system() == "Darwin"
    backend = cv2.CAP_AVFOUNDATION if is_mac else cv2.CAP_ANY

    # Try camera index 1 first (built-in Mac webcam), fallback to 0
    cap = cv2.VideoCapture(1, backend)
    if not cap.isOpened():
         cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions / device index.")

    # Request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Flush initial empty/black frames while sensor warms up
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    stabilizer = GestureStabilizer()
    icon_cache = build_icon_cache(GESTURE_EMOJI)      # empty dict if no emoji font is available
    drawing_cache = build_drawing_cache()              # empty dict if assets/drawings/ is missing files
    glasses_img = load_glasses_image()                 # None if assets/glasses.png is missing
    debug_mode = False
    mirror = True
    prev_time = time.time()

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands, mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6,
    ) as face_detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            h_px, w_px = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            hand_infos = []
            if results.multi_hand_landmarks:
                for lm_set, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame, lm_set, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    pts = np.array([[p.x * w_px, p.y * h_px, p.z * w_px] for p in lm_set.landmark])
                    label = handed.classification[0].label  # "Left" / "Right"
                    hand_infos.append(analyze_hand(pts, label))

            two_hand_gesture, per_hand_gestures = classify_frame(hand_infos)

            y_cursor = 60
            show_glasses = False
            active_gesture_labels = []
            if two_hand_gesture:
                stable = stabilizer.update("both", two_hand_gesture)
                if stable:
                    _, _, _, card_h = draw_label(frame, stable, (20, y_cursor), accent=ACCENT_TWO_HAND)
                    y_cursor += card_h + 16
                    active_gesture_labels.append(stable)
            else:
                stabilizer.update("both", None)
                seen = set()
                for handedness, label in per_hand_gestures:
                    stable = stabilizer.update(handedness, label)
                    seen.add(handedness)
                    if stable:
                        _, _, _, card_h = draw_label(frame, f"{handedness}: {stable}", (20, y_cursor))
                        y_cursor += card_h + 16
                        active_gesture_labels.append(stable)
                        if stable == GLASSES_GESTURE:
                            show_glasses = True
                for hi in hand_infos:
                    if hi.handedness not in seen:
                        stabilizer.update(hi.handedness, None)

            draw_gesture_icons(frame, icon_cache, active_gesture_labels)
            draw_drawing_panel(frame, drawing_cache, active_gesture_labels, start_y=y_cursor + 10)

            if show_glasses and glasses_img is not None:
                rgb.flags.writeable = False
                face_results = face_detector.process(rgb)
                rgb.flags.writeable = True
                if face_results.detections:
                    for detection in face_results.detections:
                        draw_glasses_image(frame, detection, glasses_img, w_px, h_px)

            if debug_mode:
                for i, hi in enumerate(hand_infos):
                    draw_debug(frame, hi, (w_px - 170, 30 + i * 150))

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_hint_bar(frame, f"FPS: {fps:.0f}", (w_px - 100, h_px - 15))
            draw_hint_bar(frame, "q: quit  d: debug  m: mirror", (20, h_px - 15))

            cv2.imshow("Hello Webcam!", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
            elif key == ord('m'):
                mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
