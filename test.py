"""Evaluate a trained MiniXception checkpoint on FER2013 PrivateTest."""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.dataloader import load_fer2013
from src.utils import (
    plot_confusion_matrix,
    print_evaluation_results,
    save_results_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MiniXception on FER2013 PrivateTest",
    )
    p.add_argument("--data", default="data/fer2013.csv", help="Path to fer2013.csv")
    p.add_argument("--ckpt", default="models/best_minixception_final.keras",
                   help="Path to model checkpoint")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Loading data from {args.data} ...")
    data = load_fer2013(args.data)
    print(f"Test samples: {data['X_test'].shape[0]}")

    print(f"Loading checkpoint from {args.ckpt} ...")
    model = keras.models.load_model(args.ckpt)

    test_loss, test_acc = model.evaluate(
        data["X_test"], data["y_test_oh"],
        batch_size=args.batch_size, verbose=0,
    )
    y_pred = np.argmax(
        model.predict(data["X_test"], batch_size=args.batch_size, verbose=0),
        axis=1,
    )

    print_evaluation_results(data["y_test"], y_pred, test_loss, test_acc)
    plot_confusion_matrix(
        data["y_test"], y_pred, test_acc,
        os.path.join(args.out_dir, "confusion_matrix.png"),
    )
    save_results_summary(
        {"test_accuracy": float(test_acc), "test_loss": float(test_loss)},
        os.path.join(args.out_dir, "results.json"),
    )

    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
