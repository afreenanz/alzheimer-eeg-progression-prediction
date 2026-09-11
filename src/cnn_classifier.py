"""Dual-output CNN: shared conv backbone -> classification + brain-age regression."""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config

logger = logging.getLogger(__name__)


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

    def train(self, X, y_class, y_age, validation_data=None, epochs=config.CNN_EPOCHS, verbose=0):
        """Train both heads jointly.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, C)
        y_class : np.ndarray, shape (N,)
            Integer class labels.
        y_age : np.ndarray, shape (N,)
            Chronological age in years (regression target).
        validation_data : tuple, optional
            (X_val, y_class_val, y_age_val).
        epochs : int
        verbose : int
            Keras verbosity level.

        Returns
        -------
        tf.keras.callbacks.History
        """
        self._ensure_model(X)
        val = None
        if validation_data is not None:
            X_val, yc_val, ya_val = validation_data
            val = (X_val, {"classification": yc_val, "regression": ya_val})
        history = self.model.fit(
            X,
            {"classification": y_class, "regression": y_age},
            validation_data=val,
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=verbose,
        )
        logger.info(
            "CNN training complete: final train loss=%.4f",
            history.history["loss"][-1],
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
            (class_probabilities [N, n_classes], predicted_age [N]).
        """
        class_probs, age_pred = self.model.predict(X, verbose=0)
        return class_probs, age_pred.reshape(-1)

    def evaluate(self, X, y_class, y_age):
        """Evaluate both heads.

        Parameters
        ----------
        X : np.ndarray
        y_class : np.ndarray
        y_age : np.ndarray

        Returns
        -------
        dict
            Keras metric name -> value.
        """
        results = self.model.evaluate(
            X,
            {"classification": y_class, "regression": y_age},
            verbose=0,
            return_dict=True,
        )
        return results

    def save(self, path):
        """Save model weights to `path` (.keras format)."""
        self.model.save(path)

    def load(self, path):
        """Load a previously saved model from `path`."""
        self.model = tf.keras.models.load_model(path)
