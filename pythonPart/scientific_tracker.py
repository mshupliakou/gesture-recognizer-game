"""
scientific_tracker.py — Moduł benchmarkowy A/B dla dwóch klasyfikatorów gestów.

Instalacja:
pip install mediapipe opencv-python numpy joblib scikit-learn

Co robi:
- Działa domyślnie jako nadajnik UDP (tryb headless, z opcją włączenia podglądu flagą --gui).
- Co 5 sekund losuje zadany gest (1 z 4) i wysyła komendę spawnu ('spawn_command', 'trial_id') do Unity.
- W każdej klatce odpytuje 2 klasyfikatory:
    1) MediaPipe GestureRecognizer (Model bazowy / referencyjny)
    2) CustomClassifier (Własny model — tymczasowo podpięty drugi MediaPipe)
- Do sterowania grą w Unity wysyła wyniki z Modelu 2 (Własnego).
- Mierzy czas pierwszej surowej detekcji (raw score >= 0.60) oraz maksymalny score w oknie 5s.
- Po każdym 5-sekundowym oknie dopisuje wiersz podsumowujący do pliku CSV.

Użycie:
  1. Tryb naukowy (Benchmark A/B obu klasyfikatorów do pliku CSV):
     python scientific_tracker.py
  
  2. Tryb gry z domyślnym modelem MediaPipe:
     python scientific_tracker.py --task
     
  3. Tryb gry z Twoim własnym, wytrenowanym modelem:
     python scientific_tracker.py --trained
"""

import argparse
import csv
import json
import random
import socket
import time
import urllib.request
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from helper_functions import normalize, to_media_pipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------- #
#  Konfiguracja
# --------------------------------------------------------------------------- #
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
BENCHMARK_DIR = Path(__file__).with_name("benchmarks")

ALLOWED_GESTURES = ["Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory"]
TRIAL_DURATION = 5.0       # Czas trwania pojedynczego okna testowego [s]
SCORE_THRESHOLD = 0.60     # Próg pewności (60%)
STABLE_FRAMES = 4          # Wygładzanie dla sterowania w Unity
CUSTOM_MODEL_PATH = Path(__file__).with_name("models") / "custom_gesture_model.joblib"


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Pobieranie modelu MediaPipe do {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Gotowe.")
    return MODEL_PATH

from mediapipe.tasks.python.components.processors import ClassifierOptions
def create_mp_recognizer(mode=vision.RunningMode.VIDEO):
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mode,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        canned_gesture_classifier_options=ClassifierOptions(
            score_threshold=0.0,
            category_allowlist=ALLOWED_GESTURES,
        ),
    )
    return vision.GestureRecognizer.create_from_options(options)


# =========================================================================== #
#  WŁASNY KLASYFIKATOR
# =========================================================================== #
class CustomClassifier:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Brak modelu: {model_path}")
        payload = joblib.load(model_path)
        self.model = payload.get("model")
        self.classes = payload.get("classes", [])
        self.feature_mode = payload.get("feature", "normalized")
        if self.model is None:
            raise ValueError(f"Plik modelu nie zawiera wytrenowanego klasyfikatora: {model_path}")

    def _build_features(self, landmarks_np: np.ndarray) -> np.ndarray:
        if self.feature_mode == "raw":
            feats = np.asarray(landmarks_np, dtype=np.float32).reshape(-1)
        else:
            feats = normalize(np.asarray(landmarks_np, dtype=np.float32))
        return feats.reshape(1, -1)

    def predict(self, mp_image, timestamp_ms: int, landmarks_np: np.ndarray | None) -> tuple[str, float]:
        if landmarks_np is None:
            return "None", 0.0
        x = self._build_features(landmarks_np)
        pred = self.model.predict(x)
        raw_label = str(pred[0]) if len(pred) else "none"

        score = 0.0
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(x)[0]
                score = float(np.max(probs))
            except Exception:
                score = 1.0
        else:
            score = 1.0

        return to_media_pipe(raw_label), score

    def close(self):
        return None


# --------------------------------------------------------------------------- #
#  Pomocnicze klasy (Stabilizacja i Zapis)
# --------------------------------------------------------------------------- #
class GestureStabilizer:
    def __init__(self, frames=STABLE_FRAMES):
        self.frames = frames
        self.stable = "None"
        self.cand = "None"
        self.count = 0

    def update(self, raw: str) -> str:
        if raw == self.cand:
            self.count += 1
        else:
            self.cand, self.count = raw, 1
        if self.cand != self.stable and self.count >= self.frames:
            self.stable = self.cand
        return self.stable

    def reset(self):
        self.stable = self.cand = "None"
        self.count = 0


class TrialManager:
    def __init__(self):
        BENCHMARK_DIR.mkdir(exist_ok=True)
        self.csv_path = BENCHMARK_DIR / time.strftime("benchmark_%Y%m%d_%H%M%S.csv")
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "trial_id", "trial_start_time_s", "target_gesture",
            "mp_detected", "mp_time_to_detect_s", "mp_first_detect_score", "mp_max_score", "mp_false_activations",
            "custom_detected", "custom_time_to_detect_s", "custom_first_detect_score", "custom_max_score", "custom_false_activations",
            "time_diff_s_mp_minus_custom", "faster_model"
        ])
        self.file.flush()
        self.trial_id = 0
        self.trial_start_t = 0.0
        self.target_gesture = "None"
        self._reset_metrics()

    def _reset_metrics(self):
        self.mp_detect_time, self.mp_detect_score, self.mp_max_score = None, None, 0.0
        self.mp_false_count, self.mp_current_false = 0, None
        self.cust_detect_time, self.cust_detect_score, self.cust_max_score = None, None, 0.0
        self.cust_false_count, self.cust_current_false = 0, None

    def start_new_trial(self, now_t: float):
        if self.trial_id > 0: self._save_current_trial()
        self.trial_id += 1
        self.trial_start_t = now_t
        self.target_gesture = random.choice(ALLOWED_GESTURES)
        self._reset_metrics()
        print(f"[Próba #{self.trial_id:02d} | t={now_t:6.1f}s] Zadany gest: {self.target_gesture}")

    def record_frame_predictions(self, now_t: float, mp_gest: str, mp_score: float, cust_gest: str, cust_score: float):
        if self.trial_id == 0: return
        elapsed = now_t - self.trial_start_t

        if mp_gest == self.target_gesture:
            if mp_score > self.mp_max_score: self.mp_max_score = mp_score
            if self.mp_detect_time is None and mp_score >= SCORE_THRESHOLD:
                self.mp_detect_time, self.mp_detect_score = elapsed, mp_score
        else:
            if mp_gest in ALLOWED_GESTURES and mp_score >= SCORE_THRESHOLD:
                if mp_gest != self.mp_current_false:
                    self.mp_false_count += 1
                    self.mp_current_false = mp_gest
            elif mp_score < SCORE_THRESHOLD or mp_gest not in ALLOWED_GESTURES:
                self.mp_current_false = None

        if cust_gest == self.target_gesture:
            if cust_score > self.cust_max_score: self.cust_max_score = cust_score
            if self.cust_detect_time is None and cust_score >= SCORE_THRESHOLD:
                self.cust_detect_time, self.cust_detect_score = elapsed, cust_score
        else:
            if cust_gest in ALLOWED_GESTURES and cust_score >= SCORE_THRESHOLD:
                if cust_gest != self.cust_current_false:
                    self.cust_false_count += 1
                    self.cust_current_false = cust_gest
            elif cust_score < SCORE_THRESHOLD or cust_gest not in ALLOWED_GESTURES:
                self.cust_current_false = None

    def _save_current_trial(self):
        mp_det, cust_det = self.mp_detect_time is not None, self.cust_detect_time is not None
        if mp_det and cust_det:
            diff = round(self.mp_detect_time - self.cust_detect_time, 4)
            faster = "Tie" if abs(diff) < 1e-4 else ("Custom" if diff > 0 else "MediaPipe")
        else:
            diff = ""
            faster = "MediaPipe_Only" if mp_det else ("Custom_Only" if cust_det else "None_Detected")

        self.writer.writerow([
            self.trial_id, round(self.trial_start_t, 3), self.target_gesture,
            int(mp_det), round(self.mp_detect_time, 4) if mp_det else "", round(self.mp_detect_score, 4) if mp_det else "", round(self.mp_max_score, 4), self.mp_false_count,
            int(cust_det), round(self.cust_detect_time, 4) if cust_det else "", round(self.cust_detect_score, 4) if cust_det else "", round(self.cust_max_score, 4), self.cust_false_count,
            diff, faster
        ])
        self.file.flush()

    def close(self):
        if self.trial_id > 0: self._save_current_trial()
        self.file.close()


def hand_metrics(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    palm_center = pts[[0, 5, 9, 13, 17]].mean(axis=0)
    hand_size = np.linalg.norm(pts[0, :2] - pts[9, :2]) + 1e-6
    pinch = float(np.linalg.norm(pts[4, :2] - pts[8, :2]) / hand_size)
    return pts, palm_center, pinch


def build_scientific_packet(hands, ts, trial_id, target_gesture):
    return {
        "t": round(ts, 3),
        "trial_id": trial_id,
        "spawn_command": target_gesture,
        "hands": [
            {
                "hand": h["hand"], "gesture": h["gesture"], "raw_gesture": h["raw_gesture"],
                "score": round(h["score"], 3), "palm": {k: round(float(v), 4) for k, v in zip("xyz", h["palm"])},
                "pinch": round(h["pinch"], 3),
                "landmarks": [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4)} for p in h["pts"]],
            } for h in hands
        ],
    }


# --------------------------------------------------------------------------- #
#  Pętla główna
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--gui", action="store_true", help="Pokaż okno podglądu")
    ap.add_argument("--task", action="store_true", help="Tryb gry: użyj modelu MediaPipe (bez benchmarku)")
    ap.add_argument("--trained", action="store_true", help="Tryb gry: użyj własnego modelu (bez benchmarku)")
    ap.add_argument("--custom-model", default=str(CUSTOM_MODEL_PATH), help="Ścieżka do pliku .joblib z wytrenowanym modelem")
    args = ap.parse_args()

    # Ochrona przed błędem w wyborze flag
    if args.task and args.trained:
        print("Wybrano jednocześnie --task i --trained. Domyślnie używam --trained.")
        args.task = False

    game_mode = args.task or args.trained

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened(): raise SystemExit("Nie udało się otworzyć kamery.")

    mp_recognizer = create_mp_recognizer()
    
    # Inicjalizacja własnego klasyfikatora tylko, jeśli jest potrzebny
    custom_classifier = None
    if not game_mode or args.trained:
        custom_classifier = CustomClassifier(Path(args.custom_model))
        
    stabilizer = GestureStabilizer()
    trials = TrialManager() if not game_mode else None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    t_start = time.monotonic()
    
    if game_mode:
        active_model = "MediaPipe Task" if args.task else "Własny Klasyfikator"
        print(f"{active_model}")
    else:
        print(f"BENCHMARK")
        print(f"Zapis statystyk CSV -> {trials.csv_path}")
        trials.start_new_trial(0.0)
        
    print(f"Wysyłanie ramek UDP -> {args.host}:{args.port}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            ts = time.monotonic() - t_start
            timestamp_ms = int(ts * 1000)

            if not game_mode and (ts - trials.trial_start_t >= TRIAL_DURATION):
                trials.start_new_trial(ts)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = mp_recognizer.recognize_for_video(mp_image, timestamp_ms)

            hands_payload = []
            if mp_result.hand_landmarks:
                lm = mp_result.hand_landmarks[0]
                label = mp_result.handedness[0][0].category_name
                pts, palm, pinch = hand_metrics(lm)

                mp_raw, mp_score = "None", 0.0
                cust_raw, cust_score = "None", 0.0

                # Obliczenia dla MediaPipe
                if not game_mode or args.task:
                    if mp_result.gestures and mp_result.gestures[0]:
                        mp_raw, mp_score = mp_result.gestures[0][0].category_name, float(mp_result.gestures[0][0].score)

                # Obliczenia dla własnego modelu
                if not game_mode or args.trained:
                    cust_raw, cust_score = custom_classifier.predict(mp_image, timestamp_ms, pts)

                # Wybór danych dla gry / benchmarku
                if game_mode:
                    active_raw = mp_raw if args.task else cust_raw
                    active_score = mp_score if args.task else cust_score
                    active_stable = stabilizer.update(active_raw)
                    
                    hands_payload.append({
                        "hand": label, "gesture": active_stable, "raw_gesture": active_raw,
                        "score": active_score, "pts": pts, "palm": palm, "pinch": pinch
                    })
                else:
                    cust_stable = stabilizer.update(cust_raw)
                    trials.record_frame_predictions(ts, mp_raw, mp_score, cust_raw, cust_score)
                    
                    hands_payload.append({
                        "hand": label, "gesture": cust_stable, "raw_gesture": cust_raw,
                        "score": cust_score, "pts": pts, "palm": palm, "pinch": pinch
                    })
            else:
                stabilizer.reset()
                if not game_mode:
                    trials.record_frame_predictions(ts, "None", 0.0, "None", 0.0)

            # Wysyłanie pakietu
            if game_mode:
                packet = build_scientific_packet(hands_payload, ts, 0, "")
            else:
                packet = build_scientific_packet(hands_payload, ts, trials.trial_id, trials.target_gesture)
                
            sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))

            if args.gui:
                if not game_mode:
                    cv2.putText(frame, f"Trial #{trials.trial_id}: {trials.target_gesture}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Gesture Tracker", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    finally:
        cap.release()
        if args.gui: cv2.destroyAllWindows()
        mp_recognizer.close()
        if custom_classifier: custom_classifier.close()
        if trials: trials.close()
        sock.close()
        if not game_mode:
            print(f"Dane zapisane w: {trials.csv_path}")

if __name__ == "__main__":
    main()