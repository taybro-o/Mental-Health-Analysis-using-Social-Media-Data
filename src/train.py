"""
Training and Evaluation Pipeline for Contextual Emotion Distribution Vectorizer (CEDV).

Optimized for:
- Local Workstation (NVIDIA RTX 2060 6GB VRAM) via FP16 Mixed Precision
- Seamless execution on Google Colab GPU runtimes
- Comprehensive multi-class and distribution evaluation metrics
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
import keras

from src.model import build_cedv_model, simplex_distribution, zscore_distribution
from src.preprocess import (
    CLASS_LABELS,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    NUM_CLASSES,
    create_tf_dataset,
    load_and_preprocess_dataframe,
    prepare_dataset_splits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CEDV_Train")


def compute_distribution_metrics(
    y_true_one_hot: np.ndarray,
    pred_simplex: np.ndarray,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute distribution-specific evaluation metrics:
    - Cross Entropy (Negative Log Likelihood)
    - Mean Squared Error against smoothed targets
    - Mean Shannon Entropy of predicted distributions (uncertainty/calibration)
    - Top-2 Accuracy (distribution nuance coverage)
    """
    clipped_probs = np.clip(pred_simplex, epsilon, 1.0)
    
    # Cross entropy: - sum(y_true * log(p))
    ce = -np.mean(np.sum(y_true_one_hot * np.log(clipped_probs), axis=-1))
    
    # MSE
    mse = float(np.mean((y_true_one_hot - pred_simplex) ** 2))
    
    # Average Shannon Entropy: - sum(p * log(p))
    entropy = -np.mean(np.sum(clipped_probs * np.log(clipped_probs), axis=-1))
    
    # Top-2 accuracy
    true_classes = np.argmax(y_true_one_hot, axis=-1)
    top2_preds = np.argsort(pred_simplex, axis=-1)[:, -2:]
    top2_correct = [true in top2 for true, top2 in zip(true_classes, top2_preds)]
    top2_acc = float(np.mean(top2_correct))

    return {
        "cross_entropy": float(ce),
        "distribution_mse": mse,
        "mean_shannon_entropy": float(entropy),
        "top2_accuracy": top2_acc,
    }


def plot_and_save_training_history(
    history: keras.callbacks.History,
    output_path: str = "reports/training_curves.png",
) -> None:
    """Plot and export training/validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history["loss"]) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history.history["loss"], "b-o", label="Training Loss")
    if "val_loss" in history.history:
        axes[0].plot(epochs, history.history["val_loss"], "r--s", label="Validation Loss")
    axes[0].set_title("Loss Trajectory (Categorical Cross-Entropy)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Accuracy curves
    if "categorical_accuracy" in history.history:
        acc_key = "categorical_accuracy"
        val_acc_key = "val_categorical_accuracy"
    elif "accuracy" in history.history:
        acc_key = "accuracy"
        val_acc_key = "val_accuracy"
    else:
        acc_key = None

    if acc_key:
        axes[1].plot(epochs, history.history[acc_key], "b-o", label="Training Accuracy")
        if val_acc_key in history.history:
            axes[1].plot(epochs, history.history[val_acc_key], "r--s", label="Validation Accuracy")
        axes[1].set_title("Accuracy Trajectory", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved training curves to %s", output_path)


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    output_path: str = "reports/confusion_matrix.png",
) -> None:
    """Generate and save normalized confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized Proportion"},
    )
    plt.title("Normalized Confusion Matrix (CEDV 8-Class)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Affective State", fontsize=12)
    plt.ylabel("True Ground Truth Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved confusion matrix heatmap to %s", output_path)


def run_training_pipeline(
    data_path: str = "data/dataset.csv",
    checkpoint_dir: str = "models/checkpoints",
    model_save_path: str = "models/saved_cedv/cedv_model.keras",
    reports_dir: str = "reports",
    vocab_size: int = 25000,
    seq_length: int = 128,
    batch_size: int = 32,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.05,
    enable_mixed_precision: bool = True,
) -> Dict[str, object]:
    """
    Executes full data loading, vectorizer adaptation, model compilation, training, and evaluation.
    """
    start_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Hardware Optimization & Mixed Precision
    if enable_mixed_precision:
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Enabled mixed_float16 precision policy.")
        except Exception as e:
            logger.warning("Could not set mixed precision: %s. Using default float32.", e)

    # 2. Dataset Ingestion and Preprocessing
    logger.info("Loading dataset from %s...", data_path)
    df = load_and_preprocess_dataframe(data_path)
    logger.info("Preprocessed dataset size: %d samples across %d classes.", len(df), df["class_id"].nunique())
    
    # 3. Train/Val/Test Stratified Splits
    train_df, val_df, test_df = prepare_dataset_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    logger.info("Partitions: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))

    # 4. Initialize and Adapt TextVectorization Layer
    logger.info("Adapting TextVectorization on training corpus (%d texts)...", len(train_df))
    vectorizer = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_length,
        name="text_vectorization",
    )
    vectorizer.adapt(train_df["clean_text"].values)
    vocab = vectorizer.get_vocabulary()
    logger.info("Vocabulary adapted. Actual vocab size: %d words.", len(vocab))

    # 5. Build tf.data Datasets
    train_ds = create_tf_dataset(
        train_df["clean_text"].values,
        train_df["class_id"].values,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = create_tf_dataset(
        val_df["clean_text"].values,
        val_df["class_id"].values,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = create_tf_dataset(
        test_df["clean_text"].values,
        test_df["class_id"].values,
        batch_size=batch_size,
        shuffle=False,
    )

    # 6. Build & Compile CEDV Model
    logger.info("Instantiating CEDV architecture...")
    model = build_cedv_model(
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_dim=128,
        lstm_units=64,
        conv_filters=128,
        dense_units=64,
        num_classes=NUM_CLASSES,
        vectorizer=vectorizer,
    )

    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=label_smoothing,
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    model.summary(print_fn=logger.info)

    # 7. Training Callbacks
    best_weights_path = os.path.join(checkpoint_dir, "best_cedv.weights.h5")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # 8. Train Model
    logger.info("Commencing training for up to %d epochs...", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 9. Save Production Model
    logger.info("Saving trained production model to %s...", model_save_path)
    model.save(model_save_path)

    # 10. Generate Training Curves Plot
    plot_and_save_training_history(history, os.path.join(reports_dir, "training_curves.png"))

    # 11. Rigorous Evaluation on Unseen Test Set
    logger.info("Running evaluation on test set (%d samples)...", len(test_df))
    test_texts = test_df["clean_text"].values
    y_true_indices = test_df["class_id"].values
    y_true_one_hot = tf.keras.utils.to_categorical(y_true_indices, num_classes=NUM_CLASSES)

    # Inference forward pass
    raw_logits = model.predict(test_ds)
    pred_simplex = simplex_distribution(raw_logits, temperature=1.0)
    y_pred_indices = np.argmax(pred_simplex, axis=-1)

    # Multi-class classification metrics
    report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])
    accuracy = float(report_dict["accuracy"])

    # Distribution evaluation metrics
    dist_metrics = compute_distribution_metrics(y_true_one_hot, pred_simplex)

    # Save Confusion Matrix Heatmap
    plot_and_save_confusion_matrix(
        y_true_indices,
        y_pred_indices,
        CLASS_LABELS,
        os.path.join(reports_dir, "confusion_matrix.png"),
    )

    duration = time.time() - start_time
    summary_report = {
        "execution_time_seconds": round(duration, 2),
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "top2_accuracy": dist_metrics["top2_accuracy"],
        "mean_shannon_entropy": dist_metrics["mean_shannon_entropy"],
        "distribution_mse": dist_metrics["distribution_mse"],
        "cross_entropy_loss": dist_metrics["cross_entropy"],
        "classification_report": report_dict,
    }

    report_path = os.path.join(reports_dir, "eval_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info("Evaluation metrics successfully written to %s", report_path)

    # Pretty print summary table to logs
    logger.info("================ EVALUATION SUMMARY ================")
    logger.info("Accuracy:             %.4f", accuracy)
    logger.info("Macro F1-Score:       %.4f", macro_f1)
    logger.info("Weighted F1-Score:    %.4f", weighted_f1)
    logger.info("Top-2 Accuracy:       %.4f", dist_metrics["top2_accuracy"])
    logger.info("Cross-Entropy Loss:   %.4f", dist_metrics["cross_entropy"])
    logger.info("Mean Shannon Entropy: %.4f", dist_metrics["mean_shannon_entropy"])
    logger.info("Total Pipeline Time:  %.1fs", duration)
    logger.info("=====================================================")

    return summary_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CEDV Model Training & Evaluation")
    parser.add_argument("--data", type=str, default="data/dataset.csv", help="Path to input dataset CSV")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
