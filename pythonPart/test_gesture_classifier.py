"""
test_gesture_classifier.py

Test your trained gesture classifier live from webcam.

Install:
    pip install mediapipe opencv-python numpy joblib scikit-learn

Run:
    python3 test_gesture_classifier.py --model models/custom_gesture_model.joblib
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_TASK_PATH = Path(__file__).with_name("gesture_recognizer.task")
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live test of custom gesture classifier")
    ap.add_argument("--model", default="models/custom_gesture_model.joblib", help="Path to saved classifier")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-score", type=float, default=0.0, help="Hide predictions below this confidence")
    return ap.parse_args()


def normalize_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected (21,3), got {pts.shape}")

    wrist = pts[0].copy()
    centered = pts - wrist

    scale = np.linalg.norm(pts[9, :2] - pts[0, :2])
    if scale < 1e-6:
        scale = 1e-6

    return (centered / scale).reshape(-1)


def create_recognizer():
    if not MODEL_TASK_PATH.exists():
        raise FileNotFoundError(f"Missing model task file: {MODEL_TASK_PATH}")

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def draw_hand(frame: np.ndarray, pts_px: list[tuple[int, int]]):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts_px[a], pts_px[b], (0, 255, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts_px):
        r = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, p, r, (255, 255, 255), -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    payload = joblib.load(model_path)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    recognizer = create_recognizer()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Failed to open camera. Try --camera 1")

    print(f"Loaded model: {model_path}")
    print("Keys: q/Esc to quit")

    t0 = time.monotonic()
    prev = t0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.monotonic()
            ts_ms = int((now - t0) * 1000)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_image, ts_ms)

            for i, lm in enumerate(result.hand_landmarks):
                handed = "?"
                if result.handedness and i < len(result.handedness) and result.handedness[i]:
                    handed = result.handedness[i][0].category_name

                landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                feat = normalize_landmarks(landmarks).reshape(1, -1)

                pred = model.predict(feat)[0]
                conf = 1.0
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feat)[0]
                    conf = float(np.max(proba))

                if conf < args.min_score:
                    continue

                pts_px = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
                draw_hand(frame, pts_px)

                x, y = pts_px[0]
                y = max(24, y - 18)
                label_text = f"{handed}: {pred} ({conf:.2f})"
                cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (50, 255, 80), 2, cv2.LINE_AA)

            cv2.putText(frame, f"{fps:4.0f} FPS", (20, 32), cv2.FONT_HERSHEY_DUPLEX,
                        0.7, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "q: quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("Custom Gesture Classifier Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    main()
