"""Dual-output CNN: shared conv backbone -> classification + brain-age regression."""

import json
import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

import config

logger = logging.getLogger(__name__)

MIN_SAMPLES_PER_CLASS_FOR_VAL_SPLIT = 4


def _build_dual_output_model(input_shape, n_classes, learning_rate, loss_weights):
    inputs = layers.Input(shape=input_shape, name="scalogram_input")

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    features = layers.GlobalAveragePooling2D()(x)
    features = layers.Dense(64, activation="relu")(features)
    features = layers.Dropout(0.3)(features)

    classification_output = layers.Dense(
        n_classes, activation="softmax", name="classification"
    )(features)
    regression_output = layers.Dense(1, activation="linear", name="regression")(
        features
    )

    model = models.Model(
        inputs=inputs, outputs=[classification_output, regression_output]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            "classification": "sparse_categorical_crossentropy",
            "regression": "mse",
        },
        loss_weights=loss_weights,
        metrics={"classification": ["accuracy"], "regression": ["mae"]},
    )
    return model


class CNNClassifier:
    """Shared-backbone CNN with classification + brain-age regression heads.

    Attributes
    ----------
    model : tf.keras.Model
        Dual-output Keras model (built lazily on first `train` call, or
        eagerly if `input_shape` is given at construction).
    learning_rate : float
    batch_size : int

    Notes
    -----
    Age normalization: the regression target (age, in raw years) and the
    classification target (cross-entropy, roughly 0-1 scale) live on very
    different numeric scales. Even with `loss_weights` down-weighting the
    regression loss, its much larger raw magnitude can still dominate the
    shared backbone's gradient updates, actively hurting the classification
    head. `train()` therefore z-score normalizes age internally (using
    only the training data passed to it -- no leakage) and `predict()`
    un-normalizes back to real years; callers never see normalized values.
    The normalization stats are persisted alongside the model in `save`/
    `load` so a reloaded model still un-normalizes correctly.
    """

    def __init__(
        self,
        input_shape=None,
        n_classes=len(config.LABEL_CLASSES[config.LABEL_MODE]),
        learning_rate=config.CNN_LEARNING_RATE,
        batch_size=config.CNN_BATCH_SIZE,
        loss_weights=None,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.loss_weights = loss_weights or config.CNN_LOSS_WEIGHTS
        self.model = None
        self.age_mean = 0.0
        self.age_std = 1.0
        if input_shape is not None:
            self.model = _build_dual_output_model(
                input_shape, n_classes, learning_rate, self.loss_weights
            )

    def _ensure_model(self, X):
        if self.model is None:
            input_shape = X.shape[1:]
            self.model = _build_dual_output_model(
                input_shape, self.n_classes, self.learning_rate, self.loss_weights
            )

    def _normalize_age(self, y_age):
        return (y_age - self.age_mean) / self.age_std

    def _denormalize_age(self, y_age_norm):
        return y_age_norm * self.age_std + self.age_mean

    def train(
        self,
        X,
        y_class,
        y_age,
        validation_data=None,
        validation_split=0.15,
        epochs=config.CNN_EPOCHS,
        early_stopping_patience=5,
        verbose=0,
    ):
        """Train both heads jointly.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, C)
        y_class : np.ndarray, shape (N,)
            Integer class labels.
        y_age : np.ndarray, shape (N,)
            Chronological age in years (regression target).
        validation_data : tuple, optional
            (X_val, y_class_val, y_age_val), already-normalized ages NOT
            required -- pass raw years, same as `y_age`. If not given, a
            stratified validation_split of the training data is carved
            out automatically (unless too few samples per class, in
            which case validation/early stopping is skipped).
        validation_split : float
            Fraction of (X, y_class, y_age) to hold out for validation
            when `validation_data` is not given.
        epochs : int
            Maximum epochs; early stopping may end training sooner.
        early_stopping_patience : int
            Epochs with no validation-loss improvement before stopping.
        verbose : int
            Keras verbosity level.

        Returns
        -------
        tf.keras.callbacks.History
        """
        self._ensure_model(X)
        self.age_mean = float(np.mean(y_age))
        self.age_std = float(np.std(y_age)) or 1.0
        y_age_norm = self._normalize_age(y_age)

        X_train, yc_train, ya_train = X, y_class, y_age_norm
        val = None
        callbacks = []

        if validation_data is not None:
            X_val, yc_val, ya_val = validation_data
            val = (X_val, {"classification": yc_val, "regression": self._normalize_age(ya_val)})
        elif validation_split and validation_split > 0:
            counts = np.bincount(y_class)
            if counts.min() >= MIN_SAMPLES_PER_CLASS_FOR_VAL_SPLIT:
                X_train, X_val, yc_train, yc_val, ya_train, ya_val = train_test_split(
                    X, y_class, y_age_norm,
                    test_size=validation_split, stratify=y_class,
                    random_state=config.RANDOM_SEED,
                )
                val = (X_val, {"classification": yc_val, "regression": ya_val})
            else:
                logger.info(
                    "Too few samples per class (%s) for a validation split; "
                    "training without early stopping.", counts,
                )

        if val is not None:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        history = self.model.fit(
            X_train,
            {"classification": yc_train, "regression": ya_train},
            validation_data=val,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
        logger.info(
            "CNN training complete: final train loss=%.4f%s",
            history.history["loss"][-1],
            f" (stopped at epoch {len(history.history['loss'])}/{epochs})"
            if len(history.history["loss"]) < epochs else "",
        )
        return history

    def predict(self, X):
        """Predict class probabilities and brain age.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, C)

        Returns
        -------
        tuple of np.ndarray
            (class_probabilities [N, n_classes], predicted_age [N] in
            real years, already un-normalized).
        """
        class_probs, age_pred_norm = self.model.predict(X, verbose=0)
        age_pred = self._denormalize_age(age_pred_norm.reshape(-1))
        return class_probs, age_pred

    def evaluate(self, X, y_class, y_age):
        """Evaluate both heads.

        Parameters
        ----------
        X : np.ndarray
        y_class : np.ndarray
        y_age : np.ndarray
            Raw years; normalized internally to match the model's trained scale.

        Returns
        -------
        dict
            Keras metric name -> value.
        """
        results = self.model.evaluate(
            X,
            {"classification": y_class, "regression": self._normalize_age(y_age)},
            verbose=0,
            return_dict=True,
        )
        return results

    def save(self, path):
        """Save model weights (.keras) plus the age-normalization sidecar."""
        self.model.save(path)
        norm_path = path + ".norm.json"
        with open(norm_path, "w") as f:
            json.dump({"age_mean": self.age_mean, "age_std": self.age_std}, f)

    def load(self, path):
        """Load a previously saved model, restoring age-normalization stats."""
        self.model = tf.keras.models.load_model(path)
        norm_path = path + ".norm.json"
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                norm = json.load(f)
            self.age_mean = norm["age_mean"]
            self.age_std = norm["age_std"]
        else:
            logger.warning(
                "No age-normalization sidecar found at %s; assuming "
                "un-normalized age (mean=0, std=1). Predictions from a "
                "model trained with normalization will be wrong.", norm_path,
            )
