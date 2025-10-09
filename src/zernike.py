# pylint: disable=no-member

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.special import factorial
from scipy.fftpack import fft2, fftshift


def astigmatism(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies an astigmatism aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the astigmatism aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="astigmatism",
    )


def coma(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a coma aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the coma aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="coma",
    )


def defocus(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a defocus aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the defocus aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="defocus",
    )

def gaussian_blur(
    image: np.ndarray,
    sigma: float = 5,
) -> np.ndarray:
    """
    Applies a Gaussian blur to the input image.

    Args:
        image (np.ndarray): The input image (grayscale or color).
        sigma (float): Standard deviation for the Gaussian kernel. Controls the amount of blur. Default: 5

    Returns:
        np.ndarray: The blurred image, normalized to the [0,1] range.

    Notes:
        For grayscale images, the blur is applied directly.
        For color images, the blur is applied to each channel separately.
    """
    image = np.asarray(image, dtype=np.float32)
    if len(image.shape) == 2:  # Grayscale image
        blurred_image = gaussian_filter(image, sigma=sigma)
    else:  # Color image
        # Apply the Gaussian filter to each channel separately
        blurred_image = np.stack(
            [
                gaussian_filter(image[:, :, channel], sigma=sigma)
                for channel in range(image.shape[2])
            ],
            axis=2,
        )
    return _normalize_image(blurred_image)


def quadrafoil(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a quadrafoil aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the quadrafoil aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="quadrafoil",
    )

def second_astigmatism(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a second-order astigmatism aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="second_astigmatism",
    )


def second_coma(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a second-order coma aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="second_coma",
    )



def second_quadrafoil(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a second-order quadrafoil aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="second_quadrafoil",
    )


def second_spherical(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a second-order spherical aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="second_spherical",
    )



def second_trefoil(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a second-order trefoil aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="second_trefoil",
    )


def third_astigmatism(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a third-order astigmatism aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="third_astigmatism",
    )

def spherical(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a spherical aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the spherical aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image, normalized to the [0,1] range.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="spherical",
    )



def trefoil(
    image: np.ndarray,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
) -> np.ndarray:
    """
    Applies a trefoil aberration to the input image using Zernike polynomials.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        aberration_amplitude (float): Amplitude of the trefoil aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450

    Returns:
        np.ndarray: The aberrated image.
    """
    return _zernike_blur(
        image=image,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
        name="trefoil",
    )


def _normalize_image(image, dtype=np.float32):
    """Normalize an image to the [0,1] range."""
    image = image.astype(dtype)
    return (image - image.min()) / (image.max() - image.min())

def _zernike_blur(
    image: np.ndarray,
    radial_order: int = 0,
    azimuthal_frequency: int = 0,
    aberration_amplitude: float = 0.0,
    pupil_diameter: float = 3,
    wavelength: int = 450,
    name: str | None = None,
) -> np.ndarray:
    """
    Applies a Zernike polynomial-based blur to the input image, simulating optical aberrations.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        radial_order (int): Radial order of the Zernike polynomial. Default: 0
        azimuthal_frequency (int): Azimuthal frequency of the Zernike polynomial. Default: 0
        aberration_amplitude (float): Amplitude of the aberration. Default: 0.0
        pupil_diameter (float): Diameter of the pupil in millimeters. Default: 3
        wavelength (int): Wavelength of light in nanometers. Default: 450
        name (str | None): Name of the aberration (e.g., "defocus", "astigmatism").
            If provided, overrides `radial_order` and `azimuthal_frequency` with preset values.
            Default: None

    Returns:
        np.ndarray: The aberrated image, normalized to the [0,1] range.

    Notes:
        If `name` is provided, the function uses preset values for `radial_order` and `azimuthal_frequency`.
        For grayscale images, the convolution is applied directly.
        For color images, the convolution is applied to each channel separately.
    """
    if name == "defocus":
        radial_order = 2
        azimuthal_frequency = 0
    elif name == "astigmatism":
        radial_order = 2
        azimuthal_frequency = 2
    elif name == "coma":
        radial_order = 3
        azimuthal_frequency = 1
    elif name == "trefoil":
        radial_order = 3
        azimuthal_frequency = 3
    elif name == "spherical":
        radial_order = 4
        azimuthal_frequency = 0
    elif name == "second_astigmatism":
        radial_order = 4
        azimuthal_frequency = 2
    elif name == "quadrafoil":
        radial_order = 4
        azimuthal_frequency = 4
    elif name == "second_coma":
        radial_order = 5
        azimuthal_frequency = 1
    elif name == "second_trefoil":
        radial_order = 5
        azimuthal_frequency = 3
    elif name == "second_spherical":
        radial_order = 6
        azimuthal_frequency = 0
    elif name == "third_astigmatism":
        radial_order = 6
        azimuthal_frequency = 2
    elif name == "second_quadrafoil":
        radial_order = 6
        azimuthal_frequency = 4

    kernel = _zernike_psf(
        radial_order=radial_order,
        azimuthal_frequency=azimuthal_frequency,
        aberration_amplitude=aberration_amplitude,
        pupil_diameter=pupil_diameter,
        wavelength=wavelength,
    )

    if len(image.shape) == 2:  # Grayscale image
        blurred = convolve(image, kernel, mode="same")
    else:  # Color image
        # Apply the convolution to each channel separately
        blurred = np.stack(
            [
                convolve(image[:, :, channel], kernel, mode="same")
                for channel in range(image.shape[2])
            ],
            axis=2,
        )
    return blurred


def _zernike_psf(
    radial_order: int,
    azimuthal_frequency: int,
    aberration_amplitude: float,
    pupil_diameter: float = 3,
    wavelength: int = 450,
):
    """
    Computes the point spread function (PSF) of an optical system with Zernike aberrations.

    Args:
        radial_order (int): The radial order of the Zernike polynomial.
        azimuthal_frequency (int): The azimuthal frequency of the Zernike polynomial.
        aberration_amplitude (float): The amplitude of the aberration in micrometers.
        pupil_diameter (float, optional): The diameter of the pupil in millimeters. Defaults to 3.
        wavelength (int, optional): The wavelength of light in nanometers. Defaults to 450.

    Returns:
        np.ndarray: The point spread function (PSF) of the optical system.
    """
    # Convert units for calculation
    aberration_amplitude *= 1e-3  # mm
    wavelength *= 1e-6  # mm
    pupil_diameter = pupil_diameter / wavelength  # number of wavelengths
    pupil_radius = pupil_diameter / 2  # number of wavelengths
    pupil_area = np.pi * pupil_radius**2  # number of wavelengths squared

    # Set up cartesian grid
    xmin, xmax = -25000, 25000
    ymin, ymax = -25000, 25000
    dx, dy = 250, 250

    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + dy, dy)

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Set up circular pupil
    circular_pupil = (np.sqrt(X**2 + Y**2) <= pupil_radius).astype(np.float32)

    # Compute Zernike polynomial
    normalization_factor = np.sqrt(
        2 * (radial_order + 1) / (1 + (azimuthal_frequency == 0))
    )

    if radial_order == 0:
        zernike_polynomial = circular_pupil
    else:
        zernike_polynomial = np.zeros_like(X, dtype=np.float64)
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X + 1e-20)
        for s in range(int(0.5 * (radial_order - abs(azimuthal_frequency)) + 1)):
            zernike_polynomial += (
                (-1) ** s
                * factorial(radial_order - s)
                * (r / pupil_radius) ** (radial_order - 2 * s)
                / (
                    factorial(s)
                    * factorial(0.5 * (radial_order + abs(azimuthal_frequency)) - s)
                    * factorial(0.5 * (radial_order - abs(azimuthal_frequency)) - s)
                )
            )
        zernike_polynomial *= (
            circular_pupil
            * normalization_factor
            * (
                (azimuthal_frequency >= 0) * np.cos(azimuthal_frequency * theta)
                - (azimuthal_frequency < 0) * np.sin(azimuthal_frequency * theta)
            )
        )

    # Compute point spread function
    point_spread_function = (
        np.abs(
            fftshift(
                fft2(
                    circular_pupil
                    * np.exp(
                        -1j
                        * 2
                        * np.pi
                        * aberration_amplitude
                        * zernike_polynomial
                        / wavelength
                    )
                )
                / pupil_area
            )
        )
        ** 2
    )
    return point_spread_function / point_spread_function.sum()