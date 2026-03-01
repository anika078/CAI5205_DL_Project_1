"""Plotting, metrics, and saving utilities."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score


DISPLAY_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def plot_training_history(history, save_path):
    """Plot accuracy and loss curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MiniXception Final Model Training History", fontsize=14)

    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.axvline(best_epoch, color="g", ls="--", alpha=0.7,
                label=f'Best epoch={best_epoch} '
                      f'({history.history["val_accuracy"][best_epoch]:.4f})')
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    best_loss_epoch = int(np.argmin(history.history["val_loss"]))
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.axvline(best_loss_epoch, color="g", ls="--", alpha=0.7,
                label=f"Best loss epoch={best_loss_epoch}")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, test_acc, save_path):
    """Plot raw and normalized confusion matrices."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix — Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix — Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.suptitle(f"MiniXception Final | Test Accuracy: {test_acc * 100:.2f}%",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def print_evaluation_results(y_true, y_pred, test_loss, test_acc):
    """Print test evaluation summary and classification report."""
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")

    print(f'\n{"=" * 55}')
    print("  MINIXCEPTION FINAL -- TEST RESULTS")
    print(f"  (PrivateTest split, n={len(y_true)})")
    print(f'{"=" * 55}')
    print(f"  Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"  Test Loss:      {test_loss:.4f}")
    print(f"  F1 (Weighted):  {f1_w:.4f}")
    print(f"  F1 (Macro):     {f1_m:.4f}")
    print(f'{"=" * 55}')
    print(classification_report(y_true, y_pred,
                                target_names=DISPLAY_LABELS, digits=4))


def save_training_history(history, save_path):
    """Export training history to JSON."""
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(save_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Training history saved to {save_path}")


def save_results_summary(results_dict, save_path):
    """Export results dictionary to JSON."""
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results summary saved to {save_path}")
