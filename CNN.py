import os
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# ============================
# Configuration
# ============================
@dataclass
class Config:
    img_height: int = 28
    img_width: int = 28
    channels: int = 1
    num_classes: int = 10
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 0.001
    model_path: str = "artifacts/cnn_mnist_model.h5"
    random_state: int = 42


cfg = Config()

# Reproducibility
tf.random.set_seed(cfg.random_state)
np.random.seed(cfg.random_state)



def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, cfg.img_height, cfg.img_width, cfg.channels) / 255.0
    X_test = X_test.reshape(-1, cfg.img_height, cfg.img_width, cfg.channels) / 255.0

    y_train = to_categorical(y_train, cfg.num_classes)
    y_test = to_categorical(y_test, cfg.num_classes)

    return X_train, X_test, y_train, y_test



def build_model() -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(cfg.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



def train_model(model, X_train, y_train):
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(cfg.model_path, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ============================
# Evaluation
# ============================
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")


# ============================
# Main Execution
# ============================
def main():
    print("Starting CNN training pipeline...")

    X_train, X_test, y_train, y_test = load_data()
    model = build_model()

    model.summary()
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print("Training completed successfully")


if __name__ == "__main__":
    main()
