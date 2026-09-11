"""Brain-age regression wrapper.

Design choice: the report's class diagram lists BrainAgeRegressor as its
own class, but architecturally the regression target shares the CNN's
convolutional features with the classification head (the "Dual-Output
CNN" in the report's design diagram -- see CNNClassifier). Rather than
train a fully separate model (which would duplicate the conv backbone and
roughly double training time, a poor trade-off given the 12-hour budget),
this class is a thin wrapper that delegates fitting/prediction to the
regression head of an already-trained CNNClassifier. This keeps the
class diagram's interface intact while avoiding a redundant model.

Chronological age from participants.tsv is used as the training target
(standard practice for brain-age models); the diagnostic "brain age gap"
signal comes from comparing predicted vs chronological age at inference
time via `compute_age_gap`, not from training on a different target.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BrainAgeRegressor:
    """Thin wrapper around a trained CNNClassifier's regression head.

    Parameters
    ----------
    cnn_classifier : src.cnn_classifier.CNNClassifier
        An already-trained (or about-to-be-trained) dual-output CNN.
    """

    def __init__(self, cnn_classifier):
        self.cnn_classifier = cnn_classifier

    def fit(self, X, y_age, y_class=None, **train_kwargs):
        """Delegate training to the shared CNN (both heads train jointly).

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, C)
        y_age : np.ndarray, shape (N,)
        y_class : np.ndarray, shape (N,), optional
            Required unless the underlying CNN has already been trained;
            the shared backbone needs both targets to train jointly.
        **train_kwargs
            Forwarded to CNNClassifier.train.

        Returns
        -------
        tf.keras.callbacks.History
        """
        if y_class is None:
            raise ValueError(
                "y_class is required: the regression head shares a "
                "backbone with the classification head and both train "
                "jointly (see module docstring)."
            )
        return self.cnn_classifier.train(X, y_class, y_age, **train_kwargs)

    def predict_age(self, X):
        """Predict brain age for a batch of scalograms.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, C)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        _, age_pred = self.cnn_classifier.predict(X)
        return age_pred

    @staticmethod
    def compute_age_gap(predicted_age, chronological_age):
        """Compute the brain-age gap (predicted - chronological).

        A positive gap indicates the model predicts an "older" brain than
        chronological age -- the diagnostic signal of interest.

        Parameters
        ----------
        predicted_age : np.ndarray or float
        chronological_age : np.ndarray or float

        Returns
        -------
        np.ndarray or float
        """
        return np.asarray(predicted_age) - np.asarray(chronological_age)
