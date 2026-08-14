#!/usr/bin/env python3
"""Train five ImageNet transfer-learning models on the Hangeul dataset.

The script preserves the supplied subject-wise train/validation/test split,
preprocesses every source image deterministically, and applies augmentation
only to training batches via ImageDataGenerator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as exc:
    raise SystemExit(
        "TensorFlow belum terpasang. Jalankan: pip install -r requirements.txt"
    ) from exc


MODEL_ORDER = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0", "xception"]
EXPECTED_SPLITS = {"train": 1216, "validation": 256, "test": 288}
EXPECTED_RESPONDENTS = {"train": 38, "validation": 8, "test": 9}
RESPONDENT_PATTERN = re.compile(r"respondent_(\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, help="Folder dataset_tf yang sudah diekstrak")
    source.add_argument("--dataset-zip", type=Path, help="ZIP dataset asli")
    parser.add_argument("--output-dir", type=Path, default=Path("training_output"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Model yang dilatih; default menjalankan seluruh model",
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--feature-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--steps-multiplier",
        type=int,
        default=1,
        help="1 = satu lintasan data per epoch; 5 = sekitar lima variasi per citra per epoch",
    )
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--overwrite-preprocessed", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch per tahap untuk uji pipeline")
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if args.feature_epochs is not None:
        config["feature_epochs"] = args.feature_epochs
    if args.fine_tune_epochs is not None:
        config["fine_tune_epochs"] = args.fine_tune_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.smoke_test:
        config["feature_epochs"] = 1
        config["fine_tune_epochs"] = 1
    config["steps_multiplier"] = args.steps_multiplier
    return config


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def safe_extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Path tidak aman di ZIP: {member.filename}")
        archive.extractall(destination)
    candidates = [p for p in destination.rglob("dataset_tf") if p.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Folder dataset_tf harus tepat satu, ditemukan {len(candidates)}")
    return candidates[0]


def audit_dataset(root: Path, output_path: Path) -> dict:
    report: dict = {"root": str(root.resolve()), "splits": {}, "overlaps": {}, "errors": []}
    all_hashes: dict[str, list[str]] = defaultdict(list)
    split_ids: dict[str, set[int]] = {}
    class_sets: list[set[str]] = []

    for split in ("train", "validation", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            report["errors"].append(f"Folder split tidak ditemukan: {split_dir}")
            continue
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        classes = [p.name for p in class_dirs]
        class_sets.append(set(classes))
        class_counts: dict[str, int] = {}
        respondent_ids: set[int] = set()
        dimensions: Counter[str] = Counter()
        modes: Counter[str] = Counter()

        for class_dir in class_dirs:
            files = sorted(class_dir.glob("*.png"))
            class_counts[class_dir.name] = len(files)
            for path in files:
                match = RESPONDENT_PATTERN.fullmatch(path.name)
                if not match:
                    report["errors"].append(f"Nama file tidak valid: {path}")
                else:
                    respondent_ids.add(int(match.group(1)))
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        dimensions[f"{image.width}x{image.height}"] += 1
                        modes[image.mode] += 1
                except Exception as exc:
                    report["errors"].append(f"Gambar rusak {path}: {exc}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                all_hashes[digest].append(str(path.relative_to(root)))

        total = sum(class_counts.values())
        split_ids[split] = respondent_ids
        report["splits"][split] = {
            "total_images": total,
            "expected_images": EXPECTED_SPLITS[split],
            "class_count": len(classes),
            "images_per_class": class_counts,
            "respondent_count": len(respondent_ids),
            "expected_respondents": EXPECTED_RESPONDENTS[split],
            "respondent_ids": sorted(respondent_ids),
            "dimensions": dict(dimensions),
            "modes": dict(modes),
        }
        if total != EXPECTED_SPLITS[split]:
            report["errors"].append(f"Jumlah {split} {total}, seharusnya {EXPECTED_SPLITS[split]}")
        if len(classes) != 32:
            report["errors"].append(f"Jumlah kelas {split} {len(classes)}, seharusnya 32")
        expected_per_class = EXPECTED_SPLITS[split] // 32
        wrong = {name: count for name, count in class_counts.items() if count != expected_per_class}
        if wrong:
            report["errors"].append(f"Kelas tidak seimbang pada {split}: {wrong}")
        if len(respondent_ids) != EXPECTED_RESPONDENTS[split]:
            report["errors"].append(
                f"Jumlah responden {split} {len(respondent_ids)}, seharusnya {EXPECTED_RESPONDENTS[split]}"
            )

    if class_sets and any(classes != class_sets[0] for classes in class_sets[1:]):
        report["errors"].append("Daftar kelas berbeda antar-subset")

    names = ["train", "validation", "test"]
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = sorted(split_ids.get(first, set()) & split_ids.get(second, set()))
            report["overlaps"][f"{first}-{second}"] = overlap
            if overlap:
                report["errors"].append(f"Data leakage {first}-{second}: {overlap}")

    duplicates = [locations for locations in all_hashes.values() if len(locations) > 1]
    report["duplicate_hash_groups"] = duplicates
    if duplicates:
        report["errors"].append(f"Ditemukan {len(duplicates)} kelompok file duplikat")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["errors"]:
        raise ValueError("Audit dataset gagal:\n- " + "\n- ".join(report["errors"]))
    return report


def crop_center_resize(source: Path, destination: Path, size: tuple[int, int]) -> None:
    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    foreground = gray < 245
    if foreground.any():
        ys, xs = np.where(foreground)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        obj = rgb.crop((left, top, right, bottom))
        margin = max(8, int(max(obj.size) * 0.12))
        side = max(obj.width, obj.height) + 2 * margin
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    else:
        canvas = ImageOps.pad(rgb, (max(rgb.size), max(rgb.size)), color="white")
    resized = canvas.resize(size, Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resized.save(destination, format="PNG", optimize=True)


def preprocess_dataset(source_root: Path, destination_root: Path, size: tuple[int, int], overwrite: bool) -> None:
    marker = destination_root / ".complete.json"
    expected = sum(EXPECTED_SPLITS.values())
    if marker.exists() and not overwrite:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("source") == str(source_root.resolve()) and state.get("count") == expected:
            print(f"Preprocessing dilewati, cache valid: {destination_root}")
            return
    if destination_root.exists() and overwrite:
        shutil.rmtree(destination_root)
    files = sorted(source_root.glob("*/*/*.png"))
    if len(files) != expected:
        raise ValueError(f"Jumlah file preprocessing {len(files)}, seharusnya {expected}")
    for index, source in enumerate(files, start=1):
        relative = source.relative_to(source_root)
        crop_center_resize(source, destination_root / relative, size)
        if index % 200 == 0 or index == len(files):
            print(f"Preprocessing {index}/{len(files)}")
    marker.write_text(
        json.dumps({"source": str(source_root.resolve()), "count": len(files), "size": size}, indent=2),
        encoding="utf-8",
    )


def architecture(name: str) -> tuple[Callable, Callable]:
    mapping = {
        "vgg16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
        "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
        "mobilenetv2": (
            tf.keras.applications.MobileNetV2,
            tf.keras.applications.mobilenet_v2.preprocess_input,
        ),
        "efficientnetb0": (
            tf.keras.applications.EfficientNetB0,
            tf.keras.applications.efficientnet.preprocess_input,
        ),
        "xception": (tf.keras.applications.Xception, tf.keras.applications.xception.preprocess_input),
    }
    return mapping[name]


def make_generators(data_root: Path, model_name: str, config: dict):
    _, preprocess_input = architecture(model_name)
    aug = config["augmentation"]
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    common = {
        "target_size": tuple(config["image_size"]),
        "batch_size": config["batch_size"],
        "class_mode": "categorical",
        "seed": config["seed"],
        "interpolation": "bilinear",
    }
    train = train_datagen.flow_from_directory(data_root / "train", shuffle=True, **common)
    validation = eval_datagen.flow_from_directory(data_root / "validation", shuffle=False, **common)
    test = eval_datagen.flow_from_directory(data_root / "test", shuffle=False, **common)
    if train.class_indices != validation.class_indices or train.class_indices != test.class_indices:
        raise ValueError("Pemetaan kelas berbeda antar-generator")
    return train, validation, test


def build_model(name: str, config: dict, weights: str) -> tuple[Model, Model]:
    constructor, _ = architecture(name)
    inputs = Input(shape=(*config["image_size"], 3), name="image")
    base = constructor(
        include_top=False,
        weights=None if weights == "none" else weights,
        input_shape=(*config["image_size"], 3),
    )
    base.trainable = False
    # Memanggil base sebagai satu nested Model membuatnya mudah ditemukan lagi
    # setelah best checkpoint dimuat sebelum tahap fine-tuning.
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dense(config["dense_units"], activation="relu", name="classifier_dense")(x)
    x = Dropout(config["dropout"], name="classifier_dropout")(x)
    outputs = Dense(config["num_classes"], activation="softmax", name="predictions")(x)
    return Model(inputs, outputs, name=f"{name}_hangeul"), base


def callbacks_for(path: Path, config: dict):
    settings = config["callbacks"]
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=settings["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=settings["reduce_lr_factor"],
            patience=settings["reduce_lr_patience"],
            min_lr=settings["min_learning_rate"],
            verbose=1,
        ),
        ModelCheckpoint(path, monitor="val_loss", save_best_only=True, verbose=1),
    ]


def merge_histories(feature_history: dict, fine_history: dict) -> pd.DataFrame:
    rows = []
    for stage, history in (("feature_extraction", feature_history), ("fine_tuning", fine_history)):
        epochs = max((len(values) for values in history.values()), default=0)
        for index in range(epochs):
            row = {"stage": stage, "stage_epoch": index + 1}
            for metric, values in history.items():
                row[metric] = values[index]
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.insert(0, "global_epoch", np.arange(1, len(frame) + 1))
    return frame


def plot_history(history: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["global_epoch"], history["accuracy"], label="Training")
    axes[0].plot(history["global_epoch"], history["val_accuracy"], label="Validation")
    axes[0].set(title=f"Akurasi {title}", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["global_epoch"], history["loss"], label="Training")
    axes[1].plot(history["global_epoch"], history["val_loss"], label="Validation")
    axes[1].set(title=f"Loss {title}", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    for axis in axes:
        transition = int((history["stage"] == "feature_extraction").sum())
        if 0 < transition < len(history):
            axis.axvline(transition + 0.5, color="black", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(model: Model, generator, class_indices: dict[str, int], output_dir: Path, model_name: str) -> dict:
    generator.reset()
    probabilities = model.predict(generator, verbose=1)
    predicted = probabilities.argmax(axis=1)
    actual = generator.classes
    index_to_class = {index: label for label, index in class_indices.items()}
    labels = list(range(len(index_to_class)))
    names = [index_to_class[index] for index in labels]

    accuracy = accuracy_score(actual, predicted)
    macro = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(actual, predicted, average="weighted", zero_division=0)
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(output_dir / "confusion_matrix.csv")
    fig, axis = plt.subplots(figsize=(18, 16))
    sns.heatmap(matrix, cmap="Blues", xticklabels=names, yticklabels=names, ax=axis, cbar=True)
    axis.set(title=f"Confusion Matrix {model_name}", xlabel="Prediksi", ylabel="Aktual")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    prediction_rows = []
    for path, truth, pred, confidence in zip(generator.filepaths, actual, predicted, probabilities.max(axis=1)):
        prediction_rows.append(
            {
                "file": str(Path(path).name),
                "actual": index_to_class[int(truth)],
                "predicted": index_to_class[int(pred)],
                "confidence": float(confidence),
                "correct": bool(truth == pred),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(output_dir / "test_predictions.csv", index=False)
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
    }


def train_one(name: str, data_root: Path, output_root: Path, config: dict, weights: str) -> dict:
    print(f"\n{'=' * 72}\nMODEL: {name.upper()}\n{'=' * 72}")
    tf.keras.backend.clear_session()
    seed_everything(config["seed"])
    model_dir = output_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    train_gen, validation_gen, test_gen = make_generators(data_root, name, config)
    class_map = train_gen.class_indices
    (model_dir / "class_indices.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model, base = build_model(name, config, weights)

    model.compile(
        optimizer=Adam(config["feature_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    feature_path = model_dir / "best_feature.keras"
    steps = len(train_gen) * config["steps_multiplier"]
    started = time.perf_counter()
    feature = model.fit(
        train_gen,
        validation_data=validation_gen,
        epochs=config["feature_epochs"],
        steps_per_epoch=steps,
        callbacks=callbacks_for(feature_path, config),
        verbose=1,
    )
    feature_seconds = time.perf_counter() - started

    model = tf.keras.models.load_model(feature_path)
    nested_models = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(nested_models) != 1:
        raise RuntimeError(f"Base model tidak unik setelah checkpoint dimuat: {nested_models}")
    base = nested_models[0]
    base.trainable = True
    unfreeze = config["fine_tune_layers"][name]
    for layer in base.layers[:-unfreeze]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(config["fine_tune_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    fine_path = model_dir / "best_finetuned.keras"
    started = time.perf_counter()
    fine = model.fit(
        train_gen,
        validation_data=validation_gen,
        epochs=config["fine_tune_epochs"],
        steps_per_epoch=steps,
        callbacks=callbacks_for(fine_path, config),
        verbose=1,
    )
    fine_seconds = time.perf_counter() - started

    best_model = tf.keras.models.load_model(fine_path)
    history = merge_histories(feature.history, fine.history)
    history.to_csv(model_dir / "history.csv", index=False)
    plot_history(history, model_dir / "training_curves.png", name.upper())
    metrics = evaluate(best_model, test_gen, class_map, model_dir, name.upper())
    metrics.update(
        {
            "model": name,
            "feature_seconds": feature_seconds,
            "fine_tune_seconds": fine_seconds,
            "total_seconds": feature_seconds + fine_seconds,
            "feature_epochs_completed": len(feature.history.get("loss", [])),
            "fine_tune_epochs_completed": len(fine.history.get("loss", [])),
            "trainable_parameters": int(sum(np.prod(v.shape) for v in best_model.trainable_weights)),
            "total_parameters": int(best_model.count_params()),
        }
    )
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args)
    seed_everything(config["seed"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.dataset_zip:
        dataset_root = safe_extract(args.dataset_zip.resolve(), output / "extracted_dataset")
    else:
        dataset_root = args.dataset_dir.resolve()
        if dataset_root.name != "dataset_tf" and (dataset_root / "dataset_tf").is_dir():
            dataset_root = dataset_root / "dataset_tf"

    audit_dataset(dataset_root, output / "dataset_audit.json")
    print("Audit dataset berhasil: subject-wise split bersih dan kelas seimbang.")

    processed = output / "preprocessed_dataset"
    if args.skip_preprocessing:
        processed = dataset_root
    else:
        preprocess_dataset(
            dataset_root,
            processed,
            tuple(config["image_size"]),
            overwrite=args.overwrite_preprocessed,
        )

    run_config = dict(config)
    run_config.update(
        {
            "models": args.models,
            "weights": args.weights,
            "dataset_root": str(dataset_root),
            "processed_root": str(processed),
            "tensorflow_version": tf.__version__,
            "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        }
    )
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    results = []
    failures = []
    for name in args.models:
        try:
            metrics = train_one(name, processed, output / "models", config, args.weights)
            results.append(metrics)
            pd.DataFrame(results).to_csv(output / "model_comparison.csv", index=False)
        except Exception as exc:
            failures.append({"model": name, "error": repr(exc)})
            (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            print(f"GAGAL {name}: {exc}", file=sys.stderr)

    if not results:
        raise SystemExit("Tidak ada model yang berhasil dilatih. Periksa failures.json.")
    frame = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    frame.to_csv(output / "model_comparison.csv", index=False)
    print("\nHASIL PERBANDINGAN\n", frame.to_string(index=False))
    if failures:
        print(f"\n{len(failures)} model gagal. Periksa failures.json.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
