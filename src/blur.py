import numpy as np
from scipy import signal


def blur(image: np.ndarray, kernel: np.ndarray, noise: float = 0.03) -> np.ndarray:
    """Blur the image and add Gaussian noise"""
    np.random.seed(111)  # Let's seed for reproducibility purposes
    blurred = np.stack(
        [signal.convolve(image[:, :, channel], kernel, mode="same") for channel in range(image.shape[2])], axis=2
    )
    blurred += np.random.normal(0, noise * blurred.max(), blurred.shape)
    return blurred
