"""
gesture_label_simple_gui.py

Proste GUI do budowy własnego klasyfikatora gestów z landmarków MediaPipe.
Uruchamiasz normalnie: python3 gesture_label_simple_gui.py

Install:
    pip install gradio mediapipe opencv-python numpy scikit-learn joblib pillow

Run:
    python3 gesture_label_simple_gui.py
"""

from __future__ import annotations

import socket
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected (21, 3), got {pts.shape}")

    wrist = pts[0].copy()
    centered = pts - wrist

    scale = np.linalg.norm(pts[9, :2] - pts[0, :2])
    if scale < 1e-6:
        scale = 1e-6

    return (centered / scale).reshape(-1)


def draw_overlay(rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(out, pts[a], pts[b], (0, 255, 255), 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 7 if i in (4, 8, 12, 16, 20) else 5
        cv2.circle(out, p, r, (255, 255, 255), -1, cv2.LINE_AA)

    return out


def create_recognizer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku modelu: {MODEL_PATH}. Umieść gesture_recognizer.task obok skryptu."
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


RECOGNIZER = create_recognizer()


def state_summary(samples: List[Dict]) -> str:
    if not samples:
        return "Próbek: 0"
    labels = [s["label"] for s in samples]
    classes = sorted(set(labels))
    counts = {c: labels.count(c) for c in classes}
    return f"Próbek: {len(samples)} | Klasy: {counts}"


def add_sample(image_path: str, label: str, samples: List[Dict]):
    if samples is None:
        samples = []
    if image_path is None:
        return None, state_summary(samples), "Dodaj obraz.", samples
    if not label or not label.strip():
        return None, state_summary(samples), "Wpisz etykietę.", samples

    bgr = cv2.imread(image_path)
    if bgr is None:
        return None, state_summary(samples), "Nie udało się wczytać obrazu.", samples

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = RECOGNIZER.recognize(mp_img)

    if not result.hand_landmarks:
        return rgb, state_summary(samples), "Nie wykryto dłoni na obrazie.", samples

    lm = result.hand_landmarks[0]
    landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    feature = normalize_landmarks(landmarks)
    preview = draw_overlay(rgb, landmarks)

    samples.append(
        {
            "label": label.strip(),
            "feature": feature,
            "landmarks": landmarks,
            "preview": preview,
        }
    )

    return preview, state_summary(samples), f"Dodano próbkę: {label.strip()}", samples


def add_samples_auto(paths, samples: List[Dict]):
    if samples is None:
        samples = []
    if not paths:
        return None, state_summary(samples), "Wybierz pliki.", build_gallery(samples), samples

    added = 0
    skipped = 0
    last_preview = None

    for path in paths:
        auto_label = label_from_image_path(path, "")
        if not auto_label:
            skipped += 1
            continue

        preview, _, status_msg, samples = add_sample(path, auto_label, samples)
        if status_msg.startswith("Dodano próbkę:"):
            added += 1
            last_preview = preview
        else:
            skipped += 1

    status = f"Dodano: {added}, pominięto: {skipped}" if skipped else f"Dodano: {added}"
    return last_preview, state_summary(samples), status, build_gallery(samples), samples


def label_from_image_path(image_path: str, current_label: str) -> str:
    if not image_path:
        return current_label
    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]
    stem = stem.replace("_", " ").replace("-", " ").strip()
    stem = re.sub(r"\s*\d+$", "", stem).strip()
    return stem or current_label


def clear_samples():
    return None, "Próbek: 0", "Wyczyszczono dataset.", []


def build_gallery(samples: List[Dict]):
    if not samples:
        return []
    show = samples[-12:]
    return [(s["preview"], s["label"]) for s in show]


def train_and_save(samples: List[Dict], out_path: str):
    if not samples or len(samples) < 4:
        return "Dodaj więcej próbek (minimum 4)."

    labels = [s["label"] for s in samples]
    classes = sorted(set(labels))
    if len(classes) < 2:
        return "Potrzebujesz co najmniej 2 różnych etykiet."

    counts = {c: labels.count(c) for c in classes}
    too_small = [c for c, n in counts.items() if n < 2]
    if too_small:
        return "Każda klasa musi mieć min. 2 próbki. Za mało: " + ", ".join(too_small)

    X = np.vstack([s["feature"] for s in samples]).astype(np.float32)
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)

    save_path = Path(out_path or "models/custom_gesture_model.joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "classes": classes,
        "feature": "normalized_21x3_flatten",
        "source": "manual_images_labels_gradio",
        "num_samples": len(samples),
    }
    joblib.dump(payload, save_path)

    return f"Model zapisany: {save_path}\n\n{report}"


def app():
    with gr.Blocks(title="Gesture Label Simple GUI") as demo:
        gr.Markdown("## Proste GUI: obraz + etykieta + trening modelu")

        samples_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                label = gr.Textbox(label="Etykieta gestu")
                image = gr.Image(label="Obraz", type="filepath")
                add_btn = gr.Button("Dodaj próbkę")
                files_auto = gr.File(label="Wiele plików (auto etykiety z nazwy)", file_count="multiple", type="filepath")
                add_many_btn = gr.Button("Dodaj wiele (auto nazwy)")
                clear_btn = gr.Button("Wyczyść")
                out_path = gr.Textbox(label="Ścieżka modelu", value="models/custom_gesture_model.joblib")
                train_btn = gr.Button("Trenuj i zapisz", variant="primary")
            with gr.Column(scale=1):
                preview = gr.Image(label="Podgląd landmarków", type="numpy")
                summary = gr.Textbox(label="Podsumowanie", value="Próbek: 0")
                status = gr.Textbox(label="Status")
                gallery = gr.Gallery(label="Ostatnie próbki", columns=3, height=300)
                report = gr.Textbox(label="Wynik treningu", lines=12)

        def on_add(img, lbl, state):
            prev, summ, stat, new_state = add_sample(img, lbl, state)
            return prev, summ, stat, build_gallery(new_state), new_state

        add_btn.click(
            on_add,
            inputs=[image, label, samples_state],
            outputs=[preview, summary, status, gallery, samples_state],
        )

        add_many_btn.click(
            add_samples_auto,
            inputs=[files_auto, samples_state],
            outputs=[preview, summary, status, gallery, samples_state],
        )

        image.change(
            label_from_image_path,
            inputs=[image, label],
            outputs=[label],
        )

        def on_clear():
            prev, summ, stat, new_state = clear_samples()
            return prev, summ, stat, [], new_state

        clear_btn.click(
            on_clear,
            inputs=[],
            outputs=[preview, summary, status, gallery, samples_state],
        )

        train_btn.click(
            train_and_save,
            inputs=[samples_state, out_path],
            outputs=[report],
        )

    return demo


def find_free_port(start: int = 7860, end: int = 7900) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 0


if __name__ == "__main__":
    port = find_free_port()
    app().launch(server_name="127.0.0.1", server_port=port, inbrowser=True)
