"""Grad-CAM explainability for the CNNClassifier's classification head."""

import logging
import os

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image

import config

logger = logging.getLogger(__name__)


def _find_last_conv_layer(model):
    """Find the last Conv2D layer name, searched from the output backward.

    Looked up dynamically (never hardcoded) since layer names can shift
    if the CNNClassifier architecture changes.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


class GradCAMVisualizer:
    """Computes and renders Grad-CAM heatmaps for a dual-output CNN.

    Attributes
    ----------
    target_layer : str or None
        Name of the conv layer to explain. Resolved dynamically from the
        model if not given explicitly.
    alpha : float
        Overlay blend transparency for the heatmap.
    """

    def __init__(self, target_layer=None, alpha=config.GRADCAM_ALPHA):
        self.target_layer = target_layer
        self.alpha = alpha

    def compute_heatmap(self, model, image, class_idx, output_name="classification"):
        """Compute a Grad-CAM heatmap for one image and target class.

        Parameters
        ----------
        model : tf.keras.Model
            Dual-output model (classification, regression).
        image : np.ndarray, shape (H, W, C)
            Single scalogram (no batch dimension).
        class_idx : int
            Index of the class to explain.
        output_name : str
            Name of the classification output layer.

        Returns
        -------
        np.ndarray, shape (H', W')
            Heatmap normalized to [0, 1], H'/W' being the target conv
            layer's spatial resolution.
        """
        layer_name = self.target_layer or _find_last_conv_layer(model)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.outputs],
        )

        image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image_batch)
            class_output_idx = model.output_names.index(output_name)
            class_score = predictions[class_output_idx][:, class_idx]

        grads = tape.gradient(class_score, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap)

        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()

    def overlay_on_image(self, image, heatmap):
        """Alpha-blend a colormapped heatmap onto the original scalogram.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, C)
            Original scalogram (C may be > 3; the mean across channels is
            used as the base grayscale image for visualization).
        heatmap : np.ndarray, shape (H', W')
            Output of `compute_heatmap`.

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype uint8
            Composite RGB overlay image.
        """
        h, w = image.shape[:2]
        base_gray = image.mean(axis=-1)
        base_gray = (base_gray - base_gray.min()) / (
            base_gray.max() - base_gray.min() + 1e-12
        )
        base_rgb = np.stack([base_gray] * 3, axis=-1)

        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = np.array(heatmap_img.resize((w, h), Image.BILINEAR)) / 255.0
        colored_heatmap = cm.jet(heatmap_resized)[:, :, :3]

        composite = (1 - self.alpha) * base_rgb + self.alpha * colored_heatmap
        composite = np.clip(composite * 255, 0, 255).astype(np.uint8)
        return composite

    def save_example(self, composite, out_name):
        """Save a composite overlay image to outputs/gradcam/.

        Parameters
        ----------
        composite : np.ndarray, shape (H, W, 3), dtype uint8
        out_name : str
            Filename (without directory) for the saved PNG.

        Returns
        -------
        str
            Full path the image was saved to.
        """
        os.makedirs(config.GRADCAM_DIR, exist_ok=True)
        out_path = os.path.join(config.GRADCAM_DIR, out_name)
        Image.fromarray(composite).save(out_path)
        logger.info("Saved Grad-CAM overlay to %s", out_path)
        return out_path
