"""Train MiniXception on FER2013."""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from src.model import build_mini_xception
from src.dataloader import load_fer2013, get_train_datagen
from src.utils import (
    plot_training_history,
    save_training_history,
    plot_confusion_matrix,
    print_evaluation_results,
    save_results_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train MiniXception on FER2013")
    p.add_argument("--data", default="data/fer2013.csv", help="Path to fer2013.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l2_reg", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--ckpt_dir", default="models")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    # Load data
    print(f"Loading data from {args.data} ...")
    data = load_fer2013(args.data)
    print(f"Train: {data['X_train'].shape} | Val: {data['X_val'].shape} "
          f"| Test: {data['X_test'].shape}")

    # Build model
    model = build_mini_xception(l2_reg=args.l2_reg, dropout_rate=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    ckpt_path = os.path.join(args.ckpt_dir, "best_minixception_final.keras")
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2,
            patience=7, min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    # Augmented training generator
    train_datagen = get_train_datagen()
    train_gen = train_datagen.flow(
        data["X_train"], data["y_train_oh"],
        batch_size=args.batch_size, seed=args.seed,
    )

    # Train
    history = model.fit(
        train_gen,
        steps_per_epoch=len(data["X_train"]) // args.batch_size,
        epochs=args.epochs,
        validation_data=(data["X_val"], data["y_val_oh"]),
        callbacks=cb,
        shuffle=True,
    )

    # Save training artifacts
    plot_training_history(history, os.path.join(args.out_dir, "training_history.png"))
    save_training_history(history, os.path.join(args.out_dir, "history.json"))

    # Print training summary
    final_train = history.history["accuracy"][-1]
    final_val = history.history["val_accuracy"][-1]
    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    print(f"\nFinal train acc:  {final_train * 100:.2f}%")
    print(f"Final val acc:    {final_val * 100:.2f}%")
    print(f"Train-val gap:    {(final_train - final_val) * 100:.2f}%")
    print(f"Best val accuracy: "
          f"{history.history['val_accuracy'][best_epoch] * 100:.2f}% "
          f"(epoch {best_epoch})")

    # Test evaluation
    test_loss, test_acc = model.evaluate(
        data["X_test"], data["y_test_oh"], verbose=0,
    )
    y_pred = np.argmax(model.predict(data["X_test"], verbose=0), axis=1)

    print_evaluation_results(data["y_test"], y_pred, test_loss, test_acc)
    plot_confusion_matrix(
        data["y_test"], y_pred, test_acc,
        os.path.join(args.out_dir, "confusion_matrix.png"),
    )
    save_results_summary(
        {"test_accuracy": float(test_acc), "test_loss": float(test_loss)},
        os.path.join(args.out_dir, "results.json"),
    )

    print(f"\nAll outputs saved to: {args.out_dir}")
    print(f"Best checkpoint:     {ckpt_path}")


if __name__ == "__main__":
    main()
