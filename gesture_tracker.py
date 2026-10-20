"""
gesture_tracker.py — independent gesture recognition module (without engine coupling).

What it does:
- shows webcam video (mirrored);
- finds hands via MediaPipe GestureRecognizer and draws a "glowing" skeleton;
- shows gesture name, confidence, FPS, and recent events log;
- records gestures to a .jsonl file (one line per event) — easy to read from any engine later;
- (optional) sends the same JSON via UDP to the game: --udp

Installation:
    pip install mediapipe opencv-python numpy

Usage:
    python gesture_tracker.py                 # just watch and record events
    python gesture_tracker.py --frames        # record every frame too (for training your own model)
    python gesture_tracker.py --udp           # + send to 127.0.0.1:5050
    python gesture_tracker.py --camera 1      # different camera

Keys:  q / Esc — quit,  r — toggle recording,  d — toggle dark background
"""

import argparse
import json
import socket
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------- #
#  Settings
# --------------------------------------------------------------------------- #
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
RECORD_DIR = Path(__file__).with_name("recordings")

STABLE_FRAMES = 4        # how many consecutive frames a gesture must hold to be considered "stable"
TRAIL_LENGTH = 22        # length of the trail behind the index finger tip

# Connections between 21 hand landmarks (standard MediaPipe topology)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm
]

# Colors (BGR) for each finger
FINGER_COLORS = {
    "thumb": (0, 140, 255),
    "index": (255, 220, 0),
    "middle": (120, 255, 80),
    "ring": (60, 220, 255),
    "pinky": (255, 80, 220),
    "palm": (230, 230, 230),
}


def connection_color(a: int, b: int):
    if (a, b) in [(0, 17), (5, 9), (9, 13), (13, 17)]:
        return FINGER_COLORS["palm"]          # palm connections
    hi = max(a, b)
    if hi <= 4:
        return FINGER_COLORS["thumb"]
    if hi <= 8:
        return FINGER_COLORS["index"]
    if hi <= 12:
        return FINGER_COLORS["middle"]
    if hi <= 16:
        return FINGER_COLORS["ring"]
    return FINGER_COLORS["pinky"]


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Downloading model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def create_recognizer():
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


# --------------------------------------------------------------------------- #
#  Hand data and "packet" for the game
# --------------------------------------------------------------------------- #
def hand_metrics(lm):
    """Palm center and 'pinch' degree (0 — fingers closed, ~1 — spread)."""
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    palm_center = pts[[0, 5, 9, 13, 17]].mean(axis=0)
    hand_size = np.linalg.norm(pts[0, :2] - pts[9, :2]) + 1e-6
    pinch = float(np.linalg.norm(pts[4, :2] - pts[8, :2]) / hand_size)
    return pts, palm_center, pinch


def build_packet(hands, ts):
    """Universal JSON — written to file and sent via UDP."""
    return {
        "t": round(ts, 3),
        "hands": [
            {
                "hand": h["hand"],
                "gesture": h["gesture"],            # stable (smoothed) gesture
                "raw_gesture": h["raw_gesture"],    # what the model returned in this frame
                "score": round(h["score"], 3),
                "palm": {k: round(float(v), 4) for k, v in zip("xyz", h["palm"])},
                "pinch": round(h["pinch"], 3),
                "landmarks": [[round(float(c), 4) for c in p] for p in h["pts"]],
            }
            for h in hands
        ],
    }


# --------------------------------------------------------------------------- #
#  Gesture Stabilizer (removes "Fist ↔ None ↔ Fist" jitter)
# --------------------------------------------------------------------------- #
class GestureStabilizer:
    def __init__(self, frames=STABLE_FRAMES):
        self.frames = frames
        self.state = {}  # hand -> {"stable": str, "cand": str, "count": int}

    def update(self, hand: str, raw: str):
        """Returns (stable_gesture, changed_from or None)."""
        s = self.state.setdefault(hand, {"stable": "None", "cand": "None", "count": 0})
        if raw == s["cand"]:
            s["count"] += 1
        else:
            s["cand"], s["count"] = raw, 1
        if s["cand"] != s["stable"] and s["count"] >= self.frames:
            prev, s["stable"] = s["stable"], s["cand"]
            return s["stable"], prev
        return s["stable"], None

    def drop(self, hand: str):
        """Hand disappeared from the frame."""
        s = self.state.get(hand)
        if s and s["stable"] != "None":
            prev = s["stable"]
            s.update(stable="None", cand="None", count=0)
            return prev
        return None


# --------------------------------------------------------------------------- #
#  Recording to file
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self, write_frames=False):
        RECORD_DIR.mkdir(exist_ok=True)
        self.path = RECORD_DIR / time.strftime("gestures_%Y%m%d_%H%M%S.jsonl")
        self.file = open(self.path, "a", encoding="utf-8")
        self.write_frames = write_frames
        self.active = True
        self.count = 0

    def toggle(self):
        self.active = not self.active

    def _write(self, obj):
        if self.active:
            self.file.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.count += 1

    def event(self, ts, hand, gesture, prev):
        self._write({"type": "event", "t": round(ts, 3), "hand": hand,
                     "gesture": gesture, "prev": prev})

    def frame(self, packet):
        if self.write_frames and packet["hands"]:
            self._write({"type": "frame", **packet})

    def close(self):
        self.file.flush()
        self.file.close()


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def draw_skeleton(frame, hands_px, trails):
    """Draw skeleton on a separate layer, blur it -> get a neon glow."""
    layer = np.zeros_like(frame)

    for h in hands_px:
        pts = h["px"]
        for a, b in CONNECTIONS:
            cv2.line(layer, pts[a], pts[b], connection_color(a, b), 4, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 9 if i in (4, 8, 12, 16, 20) else 6
            cv2.circle(layer, p, r, (255, 255, 255), -1, cv2.LINE_AA)

    # index finger trail
    for hand, trail in trails.items():
        n = len(trail)
        for i in range(1, n):
            k = i / n
            col = (int(255 * k), int(220 * k), 0)
            cv2.line(layer, trail[i - 1], trail[i], col, max(1, int(7 * k)), cv2.LINE_AA)

    glow = cv2.GaussianBlur(layer, (0, 0), 14)
    frame = cv2.add(frame, cv2.convertScaleAbs(glow, alpha=1.6))
    frame = cv2.add(frame, layer)
    return frame


def panel(frame, x, y, w, h, alpha=0.55):
    roi = frame[y:y + h, x:x + w]
    dark = np.zeros_like(roi)
    frame[y:y + h, x:x + w] = cv2.addWeighted(roi, 1 - alpha, dark, alpha, 0)


def draw_hud(frame, hands, fps, recorder, events_log):
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    # --- left panel: hand gestures ---
    panel_h = 30 + max(1, len(hands)) * 78
    panel(frame, 16, 16, 340, panel_h)
    if not hands:
        cv2.putText(frame, "no hands", (32, 66), font, 0.9, (150, 150, 150), 1, cv2.LINE_AA)
    for i, h in enumerate(hands):
        y0 = 40 + i * 78
        cv2.putText(frame, h["hand"].upper(), (32, y0), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, h["gesture"], (32, y0 + 32), font, 0.95,
                    FINGER_COLORS["index"], 2, cv2.LINE_AA)
        # confidence bar
        bx, by, bw = 32, y0 + 44, 300
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 8), (70, 70, 70), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * h["score"]), by + 8),
                      FINGER_COLORS["middle"], -1)

    # --- top right corner: FPS and REC ---
    cv2.putText(frame, f"{fps:4.0f} FPS", (W - 150, 36), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    if recorder.active:
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (W - 170, 30), 7, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.putText(frame, f"REC {recorder.count}", (W - 150, 62), font, 0.55,
                    (0, 0, 230), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "REC off", (W - 150, 62), font, 0.55, (130, 130, 130), 1, cv2.LINE_AA)

    # --- event log bottom left ---
    if events_log:
        ph = 14 + len(events_log) * 24
        panel(frame, 16, H - ph - 16, 400, ph)
        for i, line in enumerate(events_log):
            cv2.putText(frame, line, (28, H - ph + 8 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.putText(frame, "q: quit   r: record   d: dim", (W - 330, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
#  Main Loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", action="store_true", help="write every frame with 21 landmarks to file")
    ap.add_argument("--udp", action="store_true", help="send JSON via UDP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Failed to open camera. Try --camera 1")

    recognizer = create_recognizer()
    stabilizer = GestureStabilizer()
    recorder = Recorder(write_frames=args.frames)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp else None

    trails = {}                       # hand -> deque of points
    events_log = deque(maxlen=5)      # recent events for HUD
    dim = True
    t_start = time.monotonic()
    t_prev = t_start
    fps = 0.0

    print(f"Recording to: {recorder.path}")
    if args.udp:
        print(f"UDP -> {args.host}:{args.port}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror: more natural for hand control
            H, W = frame.shape[:2]

            now = time.monotonic()
            ts = now - t_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_image, int(ts * 1000))

            hands = []
            seen = set()
            for i, lm in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name  # "Left"/"Right" (already accounting for mirror)
                # frame is mirrored, so keep MediaPipe label as is
                if result.gestures and result.gestures[i]:
                    raw, score = result.gestures[i][0].category_name, result.gestures[i][0].score
                else:
                    raw, score = "None", 0.0

                if label in seen:               # two hands with same label — distinguish by index
                    label = f"{label}{i}"
                seen.add(label)

                stable, prev = stabilizer.update(label, raw)
                if prev is not None:
                    recorder.event(ts, label, stable, prev)
                    events_log.append(f"{ts:7.2f}s  {label}: {prev} -> {stable}")

                pts, palm, pinch = hand_metrics(lm)
                px = [(int(p[0] * W), int(p[1] * H)) for p in pts]
                hands.append({"hand": label, "gesture": stable, "raw_gesture": raw,
                              "score": float(score), "pts": pts, "palm": palm,
                              "pinch": pinch, "px": px})

                tr = trails.setdefault(label, deque(maxlen=TRAIL_LENGTH))
                tr.append(px[8])

            # hands that disappeared from frame
            for label in list(stabilizer.state):
                if label not in seen:
                    prev = stabilizer.drop(label)
                    if prev:
                        recorder.event(ts, label, "None", prev)
                        events_log.append(f"{ts:7.2f}s  {label}: {prev} -> None")
            for label in list(trails):
                if label not in seen:
                    del trails[label]

            # packet for file/game
            packet = build_packet(hands, ts)
            recorder.frame(packet)
            if sock and hands:
                sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))

            # visualization
            if dim:
                frame = cv2.convertScaleAbs(frame, alpha=0.55, beta=0)
            frame = draw_skeleton(frame, hands, trails)
            draw_hud(frame, hands, fps, recorder, events_log)

            cv2.imshow("Gesture Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                recorder.toggle()
            elif key == ord("d"):
                dim = not dim
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        recorder.close()
        print(f"Saved {recorder.count} records: {recorder.path}")


if __name__ == "__main__":
    main()