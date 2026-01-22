import dataclasses
import enum
from typing import Optional

from cellpose import denoise
import numpy as np
import skimage.restoration

import tgv

from .stopping import ProportionPlateauStoppingCriterion


class DeblurringMethod(enum.Enum):
    WIENER = "wiener"
    RICHARDSON_LUCY = "richardson_lucy"
    TGV = "tgv"
    CELLPOSE = "cellpose"


@dataclasses.dataclass
class CellposeConfig:
    """Configuration for Cellpose denoising"""

    def run(self, image: np.ndarray, _: Optional[np.ndarray] = None) -> np.ndarray:
        model = denoise.DenoiseModel(model_type="deblur_cyto3")
        channels = image.shape[-1]
        output = np.zeros_like(image)
        for channel in range(channels):
            output[..., channel] = model.eval(image[..., channel])[..., 0]
        return output


@dataclasses.dataclass
class WienerConfig:
    """Configuration for Wiener algorithm (from skimage)"""

    balance: float = 0.1

    def run(self, image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        output = np.zeros_like(image)
        for channel in range(image.shape[-1]):
            if self.balance < 0.0:
                output[..., channel] = skimage.restoration.unsupervised_wiener(image[..., channel], psf, clip=False)[0]
            else:
                output[..., channel] = skimage.restoration.wiener(image[..., channel], psf, self.balance, clip=False)

        return output


@dataclasses.dataclass
class RichardsonLucyConfig:
    """Configuration for Richardson-Lucy algorithm (from skimage)"""

    num_iter: int = 50

    def run(self, image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        output = np.zeros_like(image)
        for channel in range(image.shape[-1]):
            output[..., channel] = skimage.restoration.richardson_lucy(
                np.clip(image[..., channel], 0.0, 1000.0), psf, self.num_iter, clip=False
            )

        return output


@dataclasses.dataclass
class TGVConfig:
    """Configuration for our TGV approach"""

    alpha: float = 0.0005
    beta: float = 0.2
    order: int = 2
    num_iter: int = 5000
    stopping_proportion: float = 0.001  # Improvement in 100 steps

    def run(self, image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        stopping = ProportionPlateauStoppingCriterion(self.stopping_proportion, 100, 0.9)
        return tgv.tgv_deconvolution(
            image.transpose(-1, *range(image.ndim - 1))[None],
            psf[None, None],
            [self.alpha] + [self.beta] * (self.order - 1),
            padding_mode="constant",
            num_iter=self.num_iter,
            stopping=stopping,
            verbose=True,
        )[0].transpose(*range(1, image.ndim), 0)


@dataclasses.dataclass
class DeblurringConfig:
    """Deblurring configuration"""

    method: DeblurringMethod = DeblurringMethod.TGV
    cellpose: CellposeConfig = dataclasses.field(default_factory=CellposeConfig)
    wiener: WienerConfig = dataclasses.field(default_factory=WienerConfig)
    richardson_lucy: RichardsonLucyConfig = dataclasses.field(default_factory=RichardsonLucyConfig)
    tgv: TGVConfig = dataclasses.field(default_factory=TGVConfig)

    def run(self, image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Deblur the image given its assumed psf kernel

        Args:
            image (np.ndarray): Blurred image
                Shape: ([D, ]H, W, C), dtype: float
            psf (np.ndarray): Point Spread Function kernel
                Shape: ([d, ], h, w), dtype: float

        Returns:
            np.ndarray: Deblurred image
                Shape: ([D, ]H, W, C), dtype: float

        """
        if self.method == DeblurringMethod.WIENER:
            return self.wiener.run(image, psf)
        if self.method == DeblurringMethod.RICHARDSON_LUCY:
            return self.richardson_lucy.run(image, psf)
        if self.method == DeblurringMethod.TGV:
            return self.tgv.run(image, psf)
        if self.method == DeblurringMethod.CELLPOSE:
            return self.cellpose.run(image)

        raise ValueError(f"Unknown DeblurringMethod {self.method}")
