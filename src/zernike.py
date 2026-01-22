# pylint: disable=no-member

from typing import Dict, Tuple

import numpy as np
from scipy.special import factorial  # type: ignore
from scipy.fftpack import fft2, fftshift  # type: ignore


from tgv.inverse.convolve import gaussian_kernel

OPTICAL_ABERRATIONS: Dict[str, Tuple[int, int]] = {
    "defocus": (2, 0),  # Radial order, azimuthal frequency
    "astigmastism": (2, 2),
    "coma": (3, 1),
    "trefoil": (3, 3),
    "spherical": (4, 0),
    "quadrafoil": (4, 4),
    # Others: (second spherical, trefoil, ....)
}


def gaussian_psf(sigma: float = 3.0) -> np.ndarray:
    """Gaussian psf"""
    size = 2 * int(sigma * 5) + 1
    kernel = gaussian_kernel([size, size], [sigma, sigma])[0, 0]
    kernel /= kernel.sum()
    return kernel


def zernike_psf(
    radial_order: int = 2,
    azimuthal_frequency: int = 2,
    aberration_amplitude: float = 0.08,
    pupil_diameter: float = 3.0,
    wavelength: float = 450.0,
    n_planes: int = 0,
    plane_resolution: float = 0.5,
    *,
    name: str | None = None,
) -> np.ndarray:
    """
    Computes the point spread function (PSF) of an optical system with Zernike aberrations.

    Args:
        radial_order (int): The radial order of the Zernike polynomial.
        azimuthal_frequency (int): The azimuthal frequency of the Zernike polynomial.
        aberration_amplitude (float): The amplitude of the aberration in micrometers.
        pupil_diameter (float, optional): The diameter of the pupil in millimeters. Defaults to 3.
        wavelength (int, optional): The wavelength of light in nanometers. Defaults to 450.
        n_planes (int): Number of planes. 0 for 2D. 2n+1 for 3D stacks.
        plane_resolution (float): Distance between stacks. Unit: Number of wavelength.

    Returns:
        np.ndarray: The point spread function (PSF) of the optical system.
    """
    if name in OPTICAL_ABERRATIONS:
        radial_order, azimuthal_frequency = OPTICAL_ABERRATIONS[name]

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

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X + 1e-20)
    rho = r / pupil_radius

    # Set up circular pupil
    circular_pupil = (r <= pupil_radius).astype(np.float32)

    # Compute Zernike polynomial
    normalization_factor = np.sqrt(2 * (radial_order + 1) / (1 + (azimuthal_frequency == 0)))

    if radial_order == 0:
        zernike_polynomial = circular_pupil
    else:
        zernike_polynomial = np.zeros_like(X, dtype=np.float64)
        for s in range(int(0.5 * (radial_order - abs(azimuthal_frequency)) + 1)):
            zernike_polynomial += (
                (-1) ** s
                * factorial(radial_order - s)
                * rho ** (radial_order - 2 * s)
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

    # 2D
    if n_planes == 0:
        # Compute point spread function
        point_spread_function = (
            np.abs(
                fftshift(
                    fft2(
                        circular_pupil
                        * np.exp(-1j * 2 * np.pi * aberration_amplitude * zernike_polynomial / wavelength)
                    )
                    / pupil_area
                )
            )
            ** 2
        )
        return point_spread_function / point_spread_function.sum()

    # 3D
    assert n_planes % 2 == 1
    z_positions = np.linspace(-n_planes // 2, n_planes // 2, n_planes) * plane_resolution

    # Aberration phase (fixed across z)
    aberration_phase = 2 * np.pi * aberration_amplitude * zernike_polynomial / wavelength

    psf_stack = []

    for z in z_positions:
        # Defocus phase (paraxial, normalized pupil)
        defocus_phase = np.pi * z * rho**2

        pupil_field = circular_pupil * np.exp(-1j * (aberration_phase + defocus_phase))

        field = fftshift(fft2(pupil_field)) / pupil_area
        psf = np.abs(field) ** 2

        psf_stack.append(psf)

    point_spread_function = np.stack(psf_stack, axis=0)
    return point_spread_function / point_spread_function.sum()
