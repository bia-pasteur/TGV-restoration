import dataclasses
import os
import pathlib
from typing import Tuple

import dacite
import numpy as np
import pandas as pd
import skimage
import tqdm.auto as tqdm
import yaml

from .. import deblurring
from ..blur import blur
from ..zernike import zernike_psf, gaussian_psf


@dataclasses.dataclass
class ExperimentConfig:
    """Config for running experiments"""

    aberration_amplitude: float = 0.08
    noise: float = 0.03
    psf_imperfection: float = 1.0
    datasets: Tuple[str, ...] = ("Brightfield", "Fluorescence", "Phase_Contrast")
    blurs: Tuple[str, ...] = ("astigmatism", "defocus", "coma", "quadrafoil", "spherical", "trefoil", "gaussian")
    methods: Tuple[str, ...] = ("cellpose", "wiener", "rl", "tv", "tgv", "tgv3", "tgv4")
    data_folder: pathlib.Path = pathlib.Path(os.environ.get("EXPYRUN_CWD", ".")) / "Datasets"


METHODS = {
    "cellpose": deblurring.CellposeConfig(),
    "wiener": deblurring.WienerConfig(),
    "rl": deblurring.RichardsonLucyConfig(),
    "rltv": deblurring.DeconvolultionLabRLTV(
        pathlib.Path(os.environ.get("DECONVOLUTION_LAB", "DeconvolutionLab_2.jar"))
    ),
    "tv": deblurring.TGVConfig(order=1),
    "tgv": deblurring.TGVConfig(order=2),
    "tgv3": deblurring.TGVConfig(order=3),
    "tgv4": deblurring.TGVConfig(order=4),
}


def load_image(image_path: pathlib.Path, for_display=False) -> np.ndarray:
    """Load the png images in the dataset"""
    # Load image
    image = skimage.io.imread(image_path) / 255

    inactive_channels = np.all(image == 0, axis=(0, 1))
    if not for_display or (~inactive_channels).sum() != 2:
        image = image[..., ~inactive_channels]

    # Calculate the center crop window
    window = []
    for s in image.shape[:2]:
        center = s // 2
        half_size = 128  # Half of 256
        start = max(0, center - half_size)
        end = min(s, center + half_size)
        window.append(slice(start, end))

    # Apply the crop
    image = image[tuple(window)]

    # Renormalize the crop
    image -= image.min()
    image /= image.max()

    return image


def psnr(image: np.ndarray, deblurred: np.ndarray) -> float:
    return float(10 * np.log10((image**2).max() / np.mean((image - deblurred) ** 2)))


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple]))

    scores = []

    for blur_name in tqdm.tqdm(cfg.blurs):
        if blur_name == "gaussian":
            kernel = gaussian_psf(cfg.aberration_amplitude * 70)
        else:
            kernel = zernike_psf(aberration_amplitude=cfg.aberration_amplitude, name=blur_name)[70:-70, 70:-70]

        for dataset in tqdm.tqdm(cfg.datasets):
            for image_path in (cfg.data_folder / dataset).iterdir():
                image_name = image_path.stem
                image = load_image(image_path)
                blurry = blur(image, kernel, noise=cfg.noise)

                for method_name in cfg.methods:
                    deblurred = METHODS[method_name.lower()].run(blurry, kernel).clip(0, 1)
                    scores.append(
                        {
                            "Dataset": dataset,
                            "file": image_name,
                            "Blur": blur_name,
                            "method": method_name,
                            "amplitude": cfg.aberration_amplitude,
                            "noise": cfg.noise,
                            "psf_imperfection": cfg.psf_imperfection,
                            "PSNR": psnr(image, deblurred),
                        }
                    )

                    tqdm.tqdm.write(
                        f"Image {dataset}-{image_name}, with blur {blur_name}: Method {method_name} => {psnr(image, deblurred)}"
                    )

    pd.DataFrame(scores).to_csv("scores.csv", index=False)
