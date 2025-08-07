"""Data augmentation utility using Gaussian noise.

This module expands the training set by adding noisy copies of feature vectors.
Useful for regularization and improving model robustness.
"""


import numpy as np
import logging
logger = logging.getLogger(__name__)

def add_noise(X, y, n_augment=0, noise_level=0.0, random_state=None):
    """
    Artificially expands the training data by adding Gaussian noise to the features.

    Args:
        X: Feature array (numpy.ndarray)
        y: Target array (numpy.ndarray)
        n_augment: How many noisy copies to add for each original sample
        noise_level: Strength of the noise (relative to the standard deviation of each feature)
        random_state: Seed for reproducibility

    Returns:
        X_aug: Augmented feature array
        y_aug: Augmented target array
    """
    if n_augment == 0 or noise_level == 0.0:
        logger.info("No noise added (augmentation skipped).")
        return X, y

    logger.info(
        f"Augmenting data with noise: n_augment={n_augment}, "
        f"noise_level={noise_level}, random_state={random_state}, X shape={X.shape}"
    )
    rng = np.random.default_rng(random_state)
    X_aug = [X]
    y_aug = [y]
    for i in range(n_augment):
        noise = rng.normal(0, noise_level * X.std(axis=0), X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
        logger.debug(f"Added noisy augmentation #{i+1}")
    logger.info(f"Augmentation complete. Total samples: {sum(arr.shape[0] for arr in X_aug)}")
    return np.vstack(X_aug), np.hstack(y_aug)