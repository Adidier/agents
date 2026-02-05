"""
A2A LSTM Model helper

Provides a class `A2ALSTMModel` to train, save, load and predict PV output
using an LSTM. Meant to be used by higher-level agents in `a2a.algorithms`.
"""

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


class A2ALSTMModel:
    def __init__(self, time_steps: int = 24):
        self.time_steps = time_steps
        self.model = None
        self.scaler: Optional[StandardScaler] = None

    def _build_model(self, n_features: int) -> Sequential:
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.time_steps, n_features), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def _make_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.time_steps):
            X.append(data[i : i + self.time_steps])
            y.append(data[i + self.time_steps, 0])
        return np.array(X), np.array(y)

    def fit_from_csv(
        self,
        csv_path: str,
        cols: Optional[list] = None,
        epochs: int = 50,
        batch_size: int = 32,
        test_ratio: float = 0.2,
        model_out: str = "models/lstm_pv.h5",
        scaler_out: str = "models/scaler.pkl",
        callbacks: Optional[list] = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        if cols is None:
            cols = list(df.columns)[2:8] if len(df.columns) >= 8 else list(df.columns)
        df_for_training = df[cols].astype(float)

        self.scaler = StandardScaler()
        self.scaler.fit(df_for_training)
        scaled = self.scaler.transform(df_for_training)

        X, y = self._make_sequences(scaled)
        split = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = self._build_model(X_train.shape[2])

        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(scaler_out) or ".", exist_ok=True)

        default_callbacks = [
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ModelCheckpoint(model_out, save_best_only=True, monitor="val_loss"),
        ]
        cb = callbacks or default_callbacks

        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=2,
        )

        # Save scaler
        with open(scaler_out, "wb") as f:
            pickle.dump(self.scaler, f)

    def save(self, model_path: str, scaler_path: str):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not available to save")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
        self.model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, model_path: str, scaler_path: str):
        # Load without compiling to avoid deserialization errors for metrics/loss
        # (common when Keras versions differ). Recompile explicitly if needed.
        self.model = load_model(model_path, compile=False)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        try:
            # Recompile with the standard training config so `model.predict` still works fine
            # (compilation is optional for inference but safe to set here).
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        except Exception:
            # If recompilation fails, keep the uncompiled model for inference.
            pass

    def predict_next(self, df: pd.DataFrame) -> float:
        """Given a dataframe with the same columns used in training, predict next target value."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model and scaler must be loaded before prediction")
        cols = list(df.columns)[2:8] if len(df.columns) >= 8 else list(df.columns)
        df_for_training = df[cols].astype(float)
        scaled = self.scaler.transform(df_for_training)
        if len(scaled) < self.time_steps:
            raise ValueError("Not enough data to build sequence")
        seq = scaled[-self.time_steps:][np.newaxis, :, :]
        pred_scaled = self.model.predict(seq)
        # inverse transform
        dummy = np.zeros((1, scaled.shape[1]))
        dummy[0, 0] = pred_scaled[0, 0]
        inv = self.scaler.inverse_transform(dummy)
        return float(inv[0, 0])

    def predict_sequence(self, df: pd.DataFrame, steps: int = 1) -> list:
        """Predict next `steps` values autoregressively."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model and scaler must be loaded before prediction")
        cols = list(df.columns)[2:8] if len(df.columns) >= 8 else list(df.columns)
        df_for_training = df[cols].astype(float)
        scaled = self.scaler.transform(df_for_training)
        if len(scaled) < self.time_steps:
            raise ValueError("Not enough data to build sequence")
        results = []
        window = scaled[-self.time_steps:].copy()
        for _ in range(steps):
            seq = window[np.newaxis, :, :]
            pred_scaled = self.model.predict(seq)
            # append prediction to window (as first column) and roll
            new_row = np.zeros((scaled.shape[1],))
            new_row[0] = pred_scaled[0, 0]
            # for non-target columns we keep last known values
            new_row[1:] = window[-1, 1:]
            window = np.vstack([window[1:], new_row])
            dummy = np.zeros((1, scaled.shape[1]))
            dummy[0, 0] = pred_scaled[0, 0]
            inv = self.scaler.inverse_transform(dummy)
            results.append(float(inv[0, 0]))
        return results
