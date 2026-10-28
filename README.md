# TGV Image Restoration

Code for the paper: **An Arbitrarily High Order Total Generalized Variation Restoration Algorithm against Optical Aberrations in Microscopy Imaging** submitted to the IEEE ISBI 2027 conference.

Abstract: *We propose a non-blind restoration algorithm to correct strong optical aberrations in microscopy imaging. Our method features a total generalized variation regularizer of arbitrarily high order to preserve varying degrees of smoothness within the reconstructed image. We perform extensive evaluations on simulated degradations applied to microscopy images acquired through four modalities: brightfield, fluorescence, phase contrast, and 3D fluorescence imaging. Single-mode optical aberrations up to the fourth order, including astigmatism, coma, defocus, trefoil, and quadrafoil, are considered. Our method consistently outperforms the state of the art, both on simulations across varying aberrations and imaging types, and on real acquisitions with an experimentally estimated point spread function.*

This repository contains the code for the experiments presented in our paper.

## Try on your data

We have not yet integrated our work into standard bioimage analysis software (Fiji, Icy, Napari, ...). In the meantime, you can only try it out through GitHub and Python.

Here is an example script that you may run to test on your images:
```python
import numpy as np
import tifffile

from src.deblurring import TGVConfig  # The configuration for tgv deblurring

# Load your image to be restored
aberrated_image: np.ndarray = ...  # Expects a 2D/3D image with channels as a numpy array of shape (H, W, C) or (D, H, W, C)

# Load your PSF (estimated or synthetic)
psf: np.ndarray = ...  # Expects a 2D/3D numpy array of shape (H, W) or (D, H, W)  [same psf for all channels]

# Define your TGV configuration
# It sets the smoothness order of the operator used for the reconstruction.
# You may also modify here the regularization parameters
cfg = TGVConfig(order=4)

# Run the TGV restoration
restored = cfg.run(aberrated_image, psf)

# Save the restored image
tifffile.imwrite("restored.tiff", restored)
```

Coming soon: a simplified Python installation and integration as plugins in Icy/Fiji/Napari.

## Reproduce our paper

### Dataset
The dataset we use is shipped directly within the repository, in the `Datasets` folder. It comes from the `Cellpose` dataset for 2D images and from the Cell Tracking Challenge datasets for 3D images. If you use this data, please cite the appropriate source:

* 2D images: Carsen Stringer, Tim Wang, Michalis Michaelos, and Marius Pachitariu, “Cellpose: a generalist algorithm for cellular segmentation,” Nature Methods, vol. 18, no. 1, pp. 100–106, 2021
* 3D images: Martin Maška, Vladimír Ulman, David Svoboda, Pavel Matula, Petr Matula, Cristina Ederra, Ainhoa Urbiola, Tomas España, Subramanian Venkatesan, Deepak MW Balak, et al., “A benchmark for comparison of cell tracking algorithms,” Bioinformatics, vol. 30, no. 11, pp. 1609–1617, 2014.

### Install

First clone the repository:

```bash
$ git clone git@github.com:raphaelreme/TGV-Restoration.git
$ cd TGV-Restoration
```

Install requirements (we advise first that you initialize a new conda environment with Python 3.12)

```bash
$ pip install -r requirements.txt
```

> [!WARNING]
> The `tgv` library is currently hosted in a private GitHub repository and has not yet been published on PyPI (soon to come). To reproduce and test our TGV Restoration method, please contact us. We can provide the wheels!

To be able to run DeconvolutionLab, you first need to install java and download the [DeconvolutionLab software](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)


### Run the experiments

We provide a script to reproduce all the experiments at once:

```bash
python scripts/run_all.py
```


<!-- Still under review
## Cite us

If you use this work, please cite our paper:

```bibtex
@INPROCEEDINGS{tgv_restoration,
  author={Kesenci, Yekta and Reme, Raphael and Lagache, Thibault and Le Folgoc, Loic and Angelini, Elsa and Olivo-Marin, Jean-Christophe},
  title={An Arbitrarily High Order Total Generalized Variation Restoration Algorithm against Optical Aberrations in Microscopy Imaging},
}
``` -->
