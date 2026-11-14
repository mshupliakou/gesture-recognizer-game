"""
auto_capture_gestures.py

Automatycznie robi zdjęcia gestów do zbioru treningowego.

Działa tak:
- pokazujesz gest przed kamerą,
- naciskasz spację, aby zrobić zdjęcie,
- skrypt sprawdza, czy MediaPipe wykryło dłoń,
- zapisuje tylko wtedy, gdy dłoń została wykryta,
- nazwa pliku ma format: <label>_<numer>.png

Przykład:
    python3 auto_capture_gestures.py --labels pointing_up thumb_down thumb_up victory
    python3 auto_capture_gestures.py --labels open_palm peace rock_on --out-dir my_dataset
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capture hand images for gesture dataset")
    ap.add_argument(
        "--labels",
        nargs="+",
        default=["pointing_up", "thumb_down", "thumb_up", "victory"],
        help="Lista nazw gestów do zbierania, np. --labels pointing_up thumb_down",
    )
    ap.add_argument("--out-dir", default="dataset_captured", help="Folder, do którego zapisujemy zdjęcia")
    ap.add_argument("--camera", type=int, default=0, help="Indeks kamery")
    ap.add_argument("--width", type=int, default=1280, help="Szerokość kamery")
    ap.add_argument("--height", type=int, default=720, help="Wysokość kamery")
    return ap.parse_args()


def create_recognizer() -> vision.GestureRecognizer:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku modelu: {MODEL_PATH}. Upewnij się, że gesture_recognizer.task jest obok skryptu."
        )

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def next_counter(label_dir: Path) -> int:
    existing = []
    if label_dir.exists():
        for p in label_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                existing.append(p)
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = p.stem
        suffix = stem.rsplit("_", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            nums.append(int(suffix[1]))
    return max(nums, default=0) + 1


def draw_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        radius = 7 if i in (4, 8, 12, 16, 20) else 5
        cv2.circle(frame, p, radius, (255, 255, 255), -1, cv2.LINE_AA)


def save_with_recheck(out_dir: Path, label: str, frame: np.ndarray, recognizer) -> bool:
    counter = next_counter(out_dir)
    filename = out_dir / f"{label}_{counter:03d}.png"
    ok = cv2.imwrite(str(filename), frame)
    if not ok:
        print(f"[DROP] Nie udało się zapisać {filename}")
        return False

    check_frame = cv2.imread(str(filename))
    if check_frame is None:
        filename.unlink(missing_ok=True)
        print(f"[DROP] Plik nie da się odczytać: {filename}")
        return False

    rgb = cv2.cvtColor(check_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = recognizer.recognize(mp_img)
    if not result.hand_landmarks:
        filename.unlink(missing_ok=True)
        print(f"[DROP] Brak dłoni po weryfikacji -> usunięto {filename}")
        return False

    print(f"[SAVE] {filename}")
    return True


def main() -> None:
    args = parse_args()
    labels = [str(label).strip() for label in args.labels if str(label).strip()]
    if not labels:
        raise SystemExit("Wprowadź przynajmniej jedną etykietę gestu. Przykład: --labels pointing_up thumb_down")

    # Usunięcie wcześniejszego przełączania folderów, weryfikacja na wspólnym folderze
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recognizer = create_recognizer()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"Nie udało się otworzyć kamery: {args.camera}")

    current_idx = 0
    current_label = labels[current_idx]
    print(f"Dostępne etykiety: {labels}")
    print(f"Aktualna etykieta: {current_label}")
    print("Klawisze: [a]/[d] zmiana etykiety, [spacja] zrób zdjęcie, [q] wyjście")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Nie udało się odczytać klatki z kamery.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_img)

            hand_detected = bool(result.hand_landmarks)
            if hand_detected:
                draw_landmarks(frame, result.hand_landmarks[0])

            cv2.putText(
                frame,
                f"Label: {current_label}  |  Saved: {sum(1 for _ in out_dir.iterdir() if _.suffix.lower() in {'.png', '.jpg', '.jpeg'})}",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "DLOŃ WYKRYTA" if hand_detected else "BRAK DŁONI"
            cv2.putText(
                frame,
                status_text,
                (20, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "[a]/[d] label  |  [spacja] capture  |  [q] quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Capture Helper", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("a"):
                current_idx = (current_idx - 1) % len(labels)
                current_label = labels[current_idx]
                print(f"Aktualna etykieta: {current_label}")
            elif key == ord("d"):
                current_idx = (current_idx + 1) % len(labels)
                current_label = labels[current_idx]
                print(f"Aktualna etykieta: {current_label}")
            elif key == ord(" "):
                if not hand_detected:
                    print(f"[INFO] Brak dłoni dla '{current_label}'. Zdjęcie nie zostało zapisane.")
                    continue

                save_with_recheck(out_dir, current_label, frame, recognizer)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    main()
