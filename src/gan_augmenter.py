"""Conditional DCGAN (cGAN) for scalogram augmentation.

Matches the Phase 1 report's design exactly: Fig 4.2 labels the training
loop a "Conditional GAN Training loop," and Sec 2.5's literature review
cites Luo et al.'s conditional-GAN EEG augmentation as the adapted
methodology. The generator and discriminator are both conditioned on the
one-hot class label (standard cGAN, Mirza & Osindero 2014): the generator
gets label concatenated to its noise input, and the discriminator gets
the label broadcast as extra spatial channels concatenated to the image.
This lets a single GAN instance generate samples for either class on
demand, rather than training one unconditional GAN per minority class.

LIMITATION (documented, not an oversight): capped at
config.GAN_EPOCHS (default 50-100) for the 12-hour build. This is far
below convergence for a production GAN -- generated samples will be
low-fidelity (blurry, possibly mode-collapsed). The point of including it
at reduced scale is to (a) exercise the full architecture end-to-end and
(b) let the pipeline's baseline-vs-augmented comparison (see
CNNClassifier usage in pipeline.py) empirically show whether even a
weak GAN helps or hurts downstream classification -- rather than assuming
GAN augmentation is beneficial by construction.
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config

logger = logging.getLogger(__name__)


def _build_generator(latent_dim, image_size, n_channels, n_classes):
    h, w = image_size
    h0, w0 = h // 4, w // 4

    noise_input = layers.Input(shape=(latent_dim,), name="noise")
    label_input = layers.Input(shape=(n_classes,), name="label_onehot")
    x = layers.Concatenate()([noise_input, label_input])

    x = layers.Dense(h0 * w0 * 128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((h0, w0, 128))(x)

    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Conv2D(n_channels, kernel_size=3, padding="same", activation="sigmoid")(x)

    return tf.keras.Model(inputs=[noise_input, label_input], outputs=output, name="generator")


def _build_discriminator(image_size, n_channels, n_classes):
    h, w = image_size

    image_input = layers.Input(shape=(h, w, n_channels), name="image")
    label_input = layers.Input(shape=(n_classes,), name="label_onehot")

    label_map = layers.Reshape((1, 1, n_classes))(label_input)
    label_map = layers.Lambda(
        lambda t: tf.tile(t, [1, h, w, 1]), output_shape=(h, w, n_classes)
    )(label_map)
    x = layers.Concatenate(axis=-1)([image_input, label_map])

    x = layers.Conv2D(32, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=[image_input, label_input], outputs=output, name="discriminator")


class GANAugmenter:
    """Conditional shallow DCGAN over scalograms, conditioned on class label.

    Attributes
    ----------
    image_size : tuple of int
        (height, width) of scalogram images.
    n_channels : int
        Number of stacked channels per scalogram (from CWTTransformer).
    n_classes : int
        Number of diagnostic classes (2 for AD vs HC).
    latent_dim : int
        Dimensionality of the generator's noise input.
    """

    class Generator:
        """Thin factory wrapper honoring the report's class diagram."""

        @staticmethod
        def build(latent_dim, image_size, n_channels, n_classes):
            return _build_generator(latent_dim, image_size, n_channels, n_classes)

    class Discriminator:
        """Thin factory wrapper honoring the report's class diagram."""

        @staticmethod
        def build(image_size, n_channels, n_classes):
            return _build_discriminator(image_size, n_channels, n_classes)

    def __init__(
        self,
        image_size=config.IMAGE_SIZE,
        n_channels=config.N_CHANNELS_USED,
        n_classes=len(config.LABEL_CLASSES[config.LABEL_MODE]),
        latent_dim=config.GAN_LATENT_DIM,
        learning_rate=config.GAN_LEARNING_RATE,
    ):
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.generator = _build_generator(latent_dim, image_size, n_channels, n_classes)
        self.discriminator = _build_discriminator(image_size, n_channels, n_classes)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.history = {"g_loss": [], "d_loss": []}

    def _to_onehot(self, labels):
        return tf.one_hot(tf.cast(labels, tf.int32), self.n_classes)

    @tf.function
    def _train_step(self, real_images, real_labels_onehot):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as disc_tape:
            fake_images = self.generator([noise, real_labels_onehot], training=True)
            real_pred = self.discriminator([real_images, real_labels_onehot], training=True)
            fake_pred = self.discriminator([fake_images, real_labels_onehot], training=True)
            d_loss_real = self.loss_fn(tf.ones_like(real_pred) * 0.9, real_pred)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_pred), fake_pred)
            d_loss = d_loss_real + d_loss_fake
        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        with tf.GradientTape() as gen_tape:
            fake_images = self.generator([noise, real_labels_onehot], training=True)
            fake_pred = self.discriminator([fake_images, real_labels_onehot], training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_pred), fake_pred)
        gen_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        return g_loss, d_loss

    def train(self, real_scalograms, labels, epochs=config.GAN_EPOCHS, batch_size=config.GAN_BATCH_SIZE):
        """Train the conditional GAN on real scalograms of all classes.

        Parameters
        ----------
        real_scalograms : np.ndarray, shape (N, H, W, C)
            Real scalogram images spanning all classes to be modeled.
        labels : np.ndarray, shape (N,)
            Integer class label for each scalogram (0..n_classes-1).
        epochs : int
            Number of training epochs (capped for the 12-hour build).
        batch_size : int
            Mini-batch size.

        Returns
        -------
        dict
            Per-epoch generator/discriminator loss history.
        """
        n_samples = real_scalograms.shape[0]
        if n_samples < 2:
            logger.warning(
                "Only %d real sample(s) available for GAN training -- "
                "results will be unreliable. Proceeding anyway.",
                n_samples,
            )
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                (real_scalograms.astype(np.float32), labels.astype(np.int32))
            )
            .shuffle(max(n_samples, 1))
            .batch(min(batch_size, max(n_samples, 1)))
        )

        for epoch in range(epochs):
            epoch_g_losses, epoch_d_losses = [], []
            for batch_images, batch_labels in dataset:
                labels_onehot = self._to_onehot(batch_labels)
                g_loss, d_loss = self._train_step(batch_images, labels_onehot)
                epoch_g_losses.append(float(g_loss))
                epoch_d_losses.append(float(d_loss))
            mean_g = float(np.mean(epoch_g_losses)) if epoch_g_losses else float("nan")
            mean_d = float(np.mean(epoch_d_losses)) if epoch_d_losses else float("nan")
            self.history["g_loss"].append(mean_g)
            self.history["d_loss"].append(mean_d)
            if epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1:
                logger.info(
                    "GAN epoch %d/%d: g_loss=%.4f d_loss=%.4f",
                    epoch + 1,
                    epochs,
                    mean_g,
                    mean_d,
                )
                if mean_d < 1e-3:
                    logger.warning(
                        "Discriminator loss near zero at epoch %d -- possible "
                        "mode collapse / discriminator overpowering generator.",
                        epoch + 1,
                    )
        return self.history

    def generate_samples(self, n, class_label):
        """Generate n synthetic scalogram images conditioned on a class.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        class_label : int or array-like of int
            Class to condition on. Either a single int (all n samples use
            this class) or an array of length n giving a per-sample class.

        Returns
        -------
        np.ndarray, shape (n, H, W, C)
        """
        if np.isscalar(class_label):
            labels = np.full(n, class_label, dtype=np.int32)
        else:
            labels = np.asarray(class_label, dtype=np.int32)
            if len(labels) != n:
                raise ValueError("len(class_label) must equal n when class_label is array-like.")

        noise = tf.random.normal([n, self.latent_dim])
        labels_onehot = self._to_onehot(labels)
        generated = self.generator([noise, labels_onehot], training=False)
        return generated.numpy()
