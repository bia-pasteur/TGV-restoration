import pathlib
import subprocess
import tempfile

import numpy as np
import skimage
import tifffile


class DeconvolutionLabRunner:
    """Run DeconvolutionLab2 by calling the jar directly."""

    cmd = (
        "java -jar {path} Run -image file {image} -psf file {psf} -out stack {out} "
        + "-path {cwd} -norm {norm} -algorithm {algorithm} {args}"
    )

    def __init__(self, path: pathlib.Path, normalize_psf: bool = False):
        assert path.exists()
        self.path = path
        self.normalize_psf = normalize_psf

    def run(self, image: np.ndarray, psf: np.ndarray, algorithm: str, *args) -> np.ndarray:
        dim = image.ndim

        # ImageJ tiff format: TZCYXS
        if dim == 2:
            image = image[None, None, None, ..., None]
        else:
            image = image[None, :, None, ..., None]

        if psf.ndim == 2:
            psf = psf[None, None, None, ..., None]
        else:
            psf = psf[None, :, None, ..., None]

        image = image.astype(np.float32)
        psf = psf.astype(np.float32)

        with tempfile.TemporaryDirectory(prefix="DeconvLab2-python") as _cwd:
            cwd = pathlib.Path(_cwd)

            tifffile.imwrite(cwd / "image.tif", image, imagej=True)
            tifffile.imwrite(cwd / "psf.tif", psf, imagej=True)

            args_str = " ".join(str(arg) for arg in args)
            cmd = self.cmd.format(
                path=self.path,
                image="image.tif",
                psf="psf.tif",
                out="out",
                cwd=cwd,
                norm=1.0 if self.normalize_psf else "no",
                algorithm=algorithm,
                args=args_str,
            )

            print("Calling DeconvolutionLab with: ", cmd)
            subprocess.run(cmd, check=True, shell=True)

            out = skimage.io.imread(cwd / "out.tif")

        return out
