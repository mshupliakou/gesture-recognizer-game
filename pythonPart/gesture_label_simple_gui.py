"""
gesture_label_simple_gui.py

Proste GUI do budowy własnego klasyfikatora gestów z landmarków MediaPipe.
Uruchamiasz normalnie: python3 gesture_label_simple_gui.py

Install:
    pip install gradio mediapipe opencv-python numpy scikit-learn joblib pillow matplotlib

Run:
    python3 gesture_label_simple_gui.py
"""

from __future__ import annotations

import csv
import os
import re
import socket
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from helper_functions import normalize
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
TEST_SIZE = 0.2
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

BEST_RF_PARAMS = {
    "n_estimators": 200,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": -1,
}


def normalize_label_name(raw_label: str) -> str:
    if raw_label is None:
        return ""
    value = str(raw_label).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def draw_overlay(rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:  # rysowanie ręki
    out = rgb.copy()
    h, w = out.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(out, pts[a], pts[b], (0, 255, 255), 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 7 if i in (4, 8, 12, 16, 20) else 5
        cv2.circle(out, p, r, (255, 255, 255), -1, cv2.LINE_AA)

    return out


def create_recognizer():  # tworzymy recognizer dłoni z MediaPipe
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


def build_hand_augmentation_variants(bgr: np.ndarray):
    variants = [bgr]
    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    for angle in (-18, -10, 10, 18):
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            bgr,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        variants.append(rotated)
    for dx, dy in ((-12, 0), (12, 0), (0, -12), (0, 12)):
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            bgr,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        variants.append(shifted)
    return variants


def extract_landmarks_from_image(image_path: str, label: str = "", allow_augmentation: bool = True) -> List[Dict] | None:
    if not image_path:
        return None
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None

    variants = build_hand_augmentation_variants(bgr) if allow_augmentation else [bgr]
    found: List[Dict] = []

    for variant in variants:
        rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = RECOGNIZER.recognize(mp_img)
        if not result.hand_landmarks:
            continue

        lm = result.hand_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        feature = normalize(landmarks)
        preview = draw_overlay(rgb, landmarks)
        found.append({
            "label": normalize_label_name(label),
            "feature": feature,
            "landmarks": landmarks,
            "preview": preview,
            "source_path": image_path,
        })

    return found if found else None


def add_sample(image_path: str, label: str, samples: List[Dict]):
    if samples is None:
        samples = []
    if image_path is None:
        return None, state_summary(samples), "Dodaj obraz.", samples
    label = normalize_label_name(label)
    if not label or not label.strip():
        return None, state_summary(samples), "Wpisz etykietę.", samples

    sample_variants = extract_landmarks_from_image(image_path, label, allow_augmentation=False)
    if sample_variants is None:
        return None, state_summary(samples), "Nie wykryto dłoni na obrazie.", samples

    for sample in sample_variants:
        samples.append(sample)

    first_preview = sample_variants[0]["preview"]
    msg = f"Dodano próbkę: {label.strip()}"
    return first_preview, state_summary(samples), msg, samples


def add_samples_auto(paths, samples: List[Dict]):
    if samples is None:
        samples = []
    if not paths:
        return None, state_summary(samples), "Wybierz pliki.", build_gallery(samples), samples

    added = 0
    skipped = 0
    last_preview = None
    skipped_details = []

    for path in paths:
        auto_label = label_from_image_path(path, "")
        if not auto_label:
            skipped += 1
            skipped_details.append(f"{os.path.basename(path)} — brak etykiety z nazwy pliku")
            continue

        preview, _, status_msg, samples = add_sample(path, auto_label, samples)
        if status_msg.startswith("Dodano próbkę:"):
            added += 1
            last_preview = preview
        else:
            skipped += 1
            skipped_details.append(f"{os.path.basename(path)} — {status_msg}")

    if skipped:
        details = "\n".join(f"- {item}" for item in skipped_details)
        status = f"Dodano: {added}, pominięto: {skipped}\nPominięte pliki:\n{details}"
    else:
        status = f"Dodano: {added}"
    return last_preview, state_summary(samples), status, build_gallery(samples), samples


def label_from_image_path(image_path: str, current_label: str) -> str:
    if not image_path:
        return current_label
    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]
    stem = stem.replace("_", " ").replace("-", " ").strip()
    stem = re.sub(r"\s*\d+$", "", stem).strip()
    stem = normalize_label_name(stem) if stem else current_label
    return stem or current_label


def clear_samples():
    return None, "Próbek: 0", "Wyczyszczono dataset.", []


def build_gallery(samples: List[Dict]):
    if not samples:
        return []
    show = samples[-12:]
    return [(s["preview"], s["label"]) for s in show]


def make_accuracy_plot(acc_norm: float, acc_raw: float):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ["z normalize_landmarks", "bez normalize_landmarks"]
    values = [acc_norm, acc_raw]
    bars = ax.bar(labels, values, color=["#2E86DE", "#A3CB38"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Porównanie accuracy")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    return fig


def make_confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], title: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig_w = max(5.5, 1.2 * len(classes))
    fig_h = max(4.5, 1.0 * len(classes))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    return fig


def save_csv_report(rows: List[Dict], path: Path):
    fieldnames = [
        "feature_set",
        "model_name",
        "accuracy",
        "fit_time_s",
        "predict_time_s",
        "avg_predict_time_ms",
        "total_time_s",
        "support",
        "class_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_model_accuracy_plot(rows: List[Dict]):
    labels = [r["model_name"] for r in rows]
    values = [r["accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color="#2E86DE")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Porównanie modeli")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def make_model_summary_text(rows: List[Dict]):
    best = max(rows, key=lambda item: item["accuracy"])
    lines = [
        "Porównanie modeli (ten sam split):",
        *[
            (
                f"- {r['feature_set']} | {r['model_name']}: accuracy={r['accuracy']:.3f}, "
                f"czas_treningu={r['fit_time_s']:.3f}s, avg_pred={r['avg_predict_time_ms']:.3f}ms"
            )
            for r in rows
        ],
        "",
        f"Najlepszy wynik: {best['feature_set']} / {best['model_name']} | accuracy={best['accuracy']:.3f}",
    ]
    return "\n".join(lines)


def _collect_records_from_samples(samples: List[Dict]) -> List[Tuple[str, str]]:
    unique = set()
    records: List[Tuple[str, str]] = []
    for sample in samples:
        source_path = sample.get("source_path")
        label = sample.get("label")
        if not source_path or not label:
            continue
        key = (str(source_path), str(label))
        if key in unique:
            continue
        unique.add(key)
        records.append(key)
    return records


def _build_features_for_records(records: List[Tuple[str, str]], use_augmentation: bool):
    x_values = []
    y_values = []
    for image_path, label in records:
        variants = extract_landmarks_from_image(image_path, label, allow_augmentation=use_augmentation)
        if not variants:
            continue
        for variant in variants:
            x_values.append(np.asarray(variant["feature"], dtype=np.float32))
            y_values.append(variant["label"])

    if not x_values:
        return np.empty((0, 63), dtype=np.float32), np.asarray([], dtype=object)

    return np.vstack(x_values).astype(np.float32), np.asarray(y_values)


def compare_best_model_with_augmentation(samples: List[Dict], report_dir: Path):
    records = _collect_records_from_samples(samples)
    if len(records) < 4:
        return "Porównanie augmentacji pominięte: za mało danych źródłowych."

    labels = [label for _, label in records]
    classes = sorted(set(labels))
    if len(classes) < 2:
        return "Porównanie augmentacji pominięte: potrzeba min. 2 klas."

    counts = {label: labels.count(label) for label in classes}
    too_small = [label for label, count in counts.items() if count < 2]
    if too_small:
        return "Porównanie augmentacji pominięte: za mało próbek klas: " + ", ".join(too_small)

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=labels,
    )

    train_records = [records[int(i)] for i in train_idx]
    test_records = [records[int(i)] for i in test_idx]

    scenarios = [
        ("train_no_aug_test_aug", False),
        ("train_aug_test_aug", True),
    ]

    rows = []
    for scenario_name, train_aug in scenarios:
        x_train, y_train = _build_features_for_records(train_records, use_augmentation=train_aug)
        x_test, y_test = _build_features_for_records(test_records, use_augmentation=True)

        if x_train.size == 0 or x_test.size == 0:
            rows.append({
                "scenario": scenario_name,
                "accuracy": 0.0,
                "fit_time_s": 0.0,
                "predict_time_s": 0.0,
                "total_time_s": 0.0,
                "status": "empty",
            })
            continue

        model = RandomForestClassifier(**BEST_RF_PARAMS)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        predict_start = time.perf_counter()
        y_pred = model.predict(x_test)
        total_time = time.perf_counter() - start
        fit_time = predict_start - start
        predict_time = time.perf_counter() - predict_start
        acc = float(np.mean(y_pred == y_test))

        rows.append({
            "scenario": scenario_name,
            "accuracy": acc,
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
            "total_time_s": float(total_time),
            "status": "ok",
        })

        report = classification_report(y_test, y_pred, labels=classes, digits=3)
        report_path = report_dir / f"best_rf_{scenario_name}_report.txt"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(report)

        cm_fig = make_confusion_plot(y_test, y_pred, classes, f"Best RF: {scenario_name}")
        cm_fig.savefig(report_dir / f"best_rf_{scenario_name}_cm.png", dpi=200)
        plt.close(cm_fig)

    csv_path = report_dir / "best_rf_aug_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "accuracy", "fit_time_s", "predict_time_s", "total_time_s", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    scenario_labels = [row["scenario"] for row in rows]
    acc_values = [row["accuracy"] for row in rows]
    bars = ax.bar(scenario_labels, acc_values, color=["#2E86DE", "#A3CB38"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best RF: wpływ augmentacji")
    for bar, value in zip(bars, acc_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(report_dir / "best_rf_aug_comparison.png", dpi=200)
    plt.close(fig)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return "Best RF (test z augmentacją): brak poprawnych wyników (empty)."

    return (
        "Best RF (test zawsze z augmentacją):\n"
        + "\n".join(
            f"- {row['scenario']}: accuracy={row['accuracy']:.3f}, czas={row['total_time_s']:.3f}s"
            for row in ok_rows
        )
    )


def compare_best_model_vs_mediapipe(
    samples: List[Dict],
    selected_model,
    selected_feature_mode: str,
    report_dir: Path,
):
    records = _collect_records_from_samples(samples)
    if len(records) < 4:
        return "Porównanie Best vs MediaPipe pominięte: za mało danych źródłowych."

    labels = [label for _, label in records]
    classes = sorted(set(labels))
    if len(classes) < 2:
        return "Porównanie Best vs MediaPipe pominięte: potrzeba min. 2 klas."

    counts = {label: labels.count(label) for label in classes}
    too_small = [label for label, count in counts.items() if count < 2]
    if too_small:
        return "Porównanie Best vs MediaPipe pominięte: za mało próbek klas: " + ", ".join(too_small)

    indices = np.arange(len(records))
    _, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=labels,
    )
    test_records = [records[int(i)] for i in test_idx]

    y_true = []
    y_pred_custom = []
    y_pred_mp = []
    mp_total_time = 0.0
    custom_total_time = 0.0
    used_samples = 0
    compared_rows = []

    for image_path, true_label in test_records:
        bgr = cv2.imread(image_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        mp_start = time.perf_counter()
        result = RECOGNIZER.recognize(mp_img)
        mp_total_time += time.perf_counter() - mp_start

        normalized_true = normalize_label_name(true_label)
        pred_mp = "none"
        pred_custom = "none"

        if result.gestures and result.gestures[0]:
            pred_mp = normalize_label_name(result.gestures[0][0].category_name)

        if result.hand_landmarks:
            landmarks = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
            if selected_feature_mode == "raw":
                feat = landmarks.reshape(1, -1)
            else:
                feat = normalize(landmarks).reshape(1, -1)

            custom_start = time.perf_counter()
            pred_custom = normalize_label_name(str(selected_model.predict(feat)[0]))
            custom_total_time += time.perf_counter() - custom_start

        y_true.append(normalized_true)
        y_pred_mp.append(pred_mp)
        y_pred_custom.append(pred_custom)
        used_samples += 1
        compared_rows.append({
            "image_path": str(image_path),
            "true_label": normalized_true,
            "pred_custom": pred_custom,
            "pred_mediapipe": pred_mp,
        })

    if used_samples == 0:
        return "Porównanie Best vs MediaPipe pominięte: brak poprawnie wczytanych próbek testowych."

    y_true_arr = np.asarray(y_true)
    y_pred_custom_arr = np.asarray(y_pred_custom)
    y_pred_mp_arr = np.asarray(y_pred_mp)

    custom_acc = float(np.mean(y_pred_custom_arr == y_true_arr))
    mp_acc = float(np.mean(y_pred_mp_arr == y_true_arr))

    all_cm_classes = sorted(set(y_true) | set(y_pred_custom) | set(y_pred_mp))
    if not all_cm_classes:
        all_cm_classes = classes

    rows = [
        {
            "model": "best_custom",
            "accuracy": custom_acc,
            "avg_predict_time_ms": (custom_total_time / used_samples) * 1000.0,
            "samples": used_samples,
        },
        {
            "model": "mediapipe_google",
            "accuracy": mp_acc,
            "avg_predict_time_ms": (mp_total_time / used_samples) * 1000.0,
            "samples": used_samples,
        },
    ]

    csv_path = report_dir / "best_vs_mediapipe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "accuracy", "avg_predict_time_ms", "samples"])
        writer.writeheader()
        writer.writerows(rows)

    samples_csv_path = report_dir / "best_vs_mediapipe_samples.csv"
    with samples_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_label", "pred_custom", "pred_mediapipe"])
        writer.writeheader()
        writer.writerows(compared_rows)

    custom_report = classification_report(
        y_true_arr,
        y_pred_custom_arr,
        labels=all_cm_classes,
        target_names=all_cm_classes,
        digits=3,
        zero_division=0,
    )
    mp_report = classification_report(
        y_true_arr,
        y_pred_mp_arr,
        labels=all_cm_classes,
        target_names=all_cm_classes,
        digits=3,
        zero_division=0,
    )
    with (report_dir / "best_custom_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(custom_report)
    with (report_dir / "mediapipe_google_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(mp_report)

    custom_cm = make_confusion_plot(y_true_arr, y_pred_custom_arr, all_cm_classes, "Best custom model")
    custom_cm.savefig(report_dir / "best_custom_cm.png", dpi=200)
    plt.close(custom_cm)

    mp_cm = make_confusion_plot(y_true_arr, y_pred_mp_arr, all_cm_classes, "Google MediaPipe")
    mp_cm.savefig(report_dir / "mediapipe_google_cm.png", dpi=200)
    plt.close(mp_cm)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels_plot = ["Best custom", "Google MediaPipe"]
    values_plot = [custom_acc, mp_acc]
    bars = ax.bar(labels_plot, values_plot, color=["#2E86DE", "#F39C12"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best custom vs Google MediaPipe")
    for bar, value in zip(bars, values_plot):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(report_dir / "best_vs_mediapipe_accuracy.png", dpi=200)
    plt.close(fig)

    return (
        "Best custom vs Google MediaPipe:\n"
        f"- Te same obrazy testowe: {used_samples} próbek\n"
        f"- Best custom: accuracy={custom_acc:.3f}, avg_pred_ms={(custom_total_time / used_samples) * 1000.0:.3f}\n"
        f"- MediaPipe: accuracy={mp_acc:.3f}, avg_pred_ms={(mp_total_time / used_samples) * 1000.0:.3f}"
    )


def evaluate_model_configs(X: np.ndarray, y: np.ndarray, classes: List[str], feature_name: str, report_dir: Path):
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model_specs = [
        (
            "rf_200",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_300",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_500",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_depth_6",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=1,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_depth_none",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_min_leaf_2",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_max_features_sqrt",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "rf_max_features_log2",
            RandomForestClassifier(
                n_estimators=300,
                max_features="log2",
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]

    rows: List[Dict] = []
    best_model_name = ""
    best_acc = -1.0
    best_y_pred = None
    best_model = None

    for model_name, model in model_specs:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        predict_start = time.perf_counter()
        y_pred = model.predict(X_test)
        total_time = time.perf_counter() - start
        fit_time = predict_start - start
        predict_time = time.perf_counter() - predict_start
        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
            best_y_pred = y_pred
            best_model = model

        rows.append({
            "feature_set": feature_name,
            "model_name": model_name,
            "accuracy": float(acc),
            "fit_time_s": float(fit_time),
            "predict_time_s": float(predict_time),
            "avg_predict_time_ms": float((predict_time / len(y_test)) * 1000.0) if len(y_test) else 0.0,
            "total_time_s": float(total_time),
            "support": int(len(y_test)),
            "class_count": int(len(classes)),
        })

        report_text = classification_report(y_test, y_pred, target_names=classes, digits=3)
        report_path = report_dir / f"{feature_name}_{model_name}_report.txt"
        with report_path.open("w", encoding="utf-8") as fh:
            fh.write(report_text)

        cm_fig = make_confusion_plot(y_test, y_pred, classes, f"Macierz pomyłek: {feature_name} / {model_name}")
        cm_fig.savefig(report_dir / f"{feature_name}_{model_name}_cm.png", dpi=200)
        plt.close(cm_fig)

    summary_path = report_dir / f"{feature_name}_summary.csv"
    save_csv_report(rows, summary_path)

    if best_y_pred is not None:
        best_cm = make_confusion_plot(y_test, best_y_pred, classes, f"Najlepsza macierz pomyłek: {feature_name}")
        best_cm.savefig(report_dir / f"{feature_name}_best_cm.png", dpi=200)
        plt.close(best_cm)

    return rows, best_model_name, best_model, best_y_pred, y_test


def train_and_save(samples: List[Dict], out_path: str):
    if not samples or len(samples) < 4:
        return "Dodaj więcej próbek (minimum 4).", None, None, None

    labels = [s["label"] for s in samples]
    classes = sorted(set(labels))
    if len(classes) < 2:
        return "Potrzebujesz co najmniej 2 różnych etykiet.", None, None, None

    counts = {c: labels.count(c) for c in classes}
    too_small = [c for c, n in counts.items() if n < 2]
    if too_small:
        return "Każda klasa musi mieć min. 2 próbki. Za mało: " + ", ".join(too_small), None, None, None

    X_norm = np.vstack([s["feature"] for s in samples]).astype(np.float32)
    X_raw = np.vstack([
        np.asarray(s["landmarks"], dtype=np.float32).reshape(-1)
        for s in samples
    ]).astype(np.float32)
    y = np.asarray(labels)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y,
    )

    report_dir = Path("results") / f"gesture_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)

    norm_rows, norm_best_name, norm_best_model, y_pred_norm, y_test_norm = evaluate_model_configs(X_norm, y, classes, "normalized", report_dir)
    raw_rows, raw_best_name, raw_best_model, y_pred_raw, y_test_raw = evaluate_model_configs(X_raw, y, classes, "raw", report_dir)

    all_rows = norm_rows + raw_rows
    save_csv_report(all_rows, report_dir / "combined_summary.csv")

    acc_plot = make_model_accuracy_plot(all_rows)
    acc_plot.savefig(report_dir / "accuracy_plot.png", dpi=200)

    save_path = Path(out_path or "models/custom_gesture_model.joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_norm = max(norm_rows, key=lambda r: r["accuracy"])
    best_raw = max(raw_rows, key=lambda r: r["accuracy"])

    selected_feature_mode = "normalized"
    selected_model_name = norm_best_name
    selected_model = norm_best_model
    if best_raw["accuracy"] > best_norm["accuracy"]:
        selected_feature_mode = "raw"
        selected_model_name = raw_best_name
        selected_model = raw_best_model

    payload = {
        "model": selected_model,
        "classes": classes,
        "feature": selected_feature_mode,
        "selected_model_name": selected_model_name,
        "source": "manual_images_labels_gradio",
        "num_samples": len(samples),
        "report_dir": str(report_dir),
        "best_model_normalized": best_norm,
        "best_model_raw": best_raw,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

    joblib.dump(payload, save_path)

    summary_text = (
        f"Model zapisany: {save_path}\n"
        f"Raport: {report_dir}\n\n"
        f"{make_model_summary_text(all_rows)}\n\n"
        f"Najlepszy model (z normalize_landmarks): {best_norm['model_name']} | accuracy={best_norm['accuracy']:.3f}\n"
        f"Najlepszy model (bez normalize_landmarks): {best_raw['model_name']} | accuracy={best_raw['accuracy']:.3f}\n"
        f"Wyeksportowany model: {selected_model_name} | feature={selected_feature_mode}"
    )

    aug_summary = compare_best_model_with_augmentation(samples, report_dir)
    summary_text = summary_text + "\n\n" + aug_summary

    mp_summary = compare_best_model_vs_mediapipe(samples, selected_model, selected_feature_mode, report_dir)
    summary_text = summary_text + "\n\n" + mp_summary

    cm_norm_plot = make_confusion_plot(y_test_norm, y_pred_norm, classes, "Macierz pomyłek (z normalize_landmarks)") if y_pred_norm is not None else make_confusion_plot(y, y, classes, "Macierz pomyłek (z normalize_landmarks)")
    cm_raw_plot = make_confusion_plot(y_test_raw, y_pred_raw, classes, "Macierz pomyłek (bez normalize_landmarks)") if y_pred_raw is not None else make_confusion_plot(y, y, classes, "Macierz pomyłek (bez normalize_landmarks)")

    cm_norm_plot.savefig(report_dir / "cm_norm_best.png", dpi=200)
    cm_raw_plot.savefig(report_dir / "cm_raw_best.png", dpi=200)
    plt.close(cm_norm_plot)
    plt.close(cm_raw_plot)

    return summary_text, acc_plot, cm_norm_plot, cm_raw_plot


def app():
    with gr.Blocks(title="Gesture Label Simple GUI") as demo:
        gr.Markdown("## Proste GUI: wiele obrazów + trening modelu")

        samples_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                files_auto = gr.File(label="Wiele plików (auto etykiety z nazwy)", file_count="multiple", type="filepath")
                add_many_btn = gr.Button("Dodaj wiele (auto nazwy)")
                clear_btn = gr.Button("Wyczyść")
                out_path = gr.Textbox(label="Ścieżka modelu", value="models/custom_gesture_model.joblib")
                train_btn = gr.Button("Trenuj, testuj i zapisz", variant="primary")
            with gr.Column(scale=1):
                preview = gr.Image(label="Podgląd landmarków", type="numpy")
                summary = gr.Textbox(label="Podsumowanie", value="Próbek: 0")
                status = gr.Textbox(label="Status")
                gallery = gr.Gallery(label="Ostatnie próbki", columns=3, height=300)
                report = gr.Textbox(label="Wynik treningu", lines=12)
                plot_acc = gr.Plot(label="Wykres accuracy")
                plot_cm_norm = gr.Plot(label="Macierz pomyłek (z normalizacją)")
                plot_cm_raw = gr.Plot(label="Macierz pomyłek (bez normalizacji)")

        add_many_btn.click(
            add_samples_auto,
            inputs=[files_auto, samples_state],
            outputs=[preview, summary, status, gallery, samples_state],
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
            outputs=[report, plot_acc, plot_cm_norm, plot_cm_raw],
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
