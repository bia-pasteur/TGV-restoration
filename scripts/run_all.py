# TODO: Yekta Call expyrun for running on all datasets with all parameters

import subprocess

# Run main experiment
subprocess.run("expyrun configs/main.yml".split())


# For aberrations
methods = "cellpose,wiener,rl,tgv4"
blurs = "astigmatism,coma"
for aberration in [0.02, 0.04, 0.06, 0.10]:  # Note 0.08 is already done in main
    subprocess.run(
        f"expyrun configs/main.yml --aberration_amplitude {aberration} --methods {methods} --blurs {blurs}".split()
    )


# For PSF imperfections
for psf_imperfection in [0.9, 0.8, 0.7, 0.6]:  # Note 1 is already done in main
    subprocess.run(
        f"expyrun configs/main.yml --psf_imperfection {psf_imperfection} --methods {methods} --blurs {blurs}".split()
    )

# For Noise stds
for std in [0, 0.01, 0.02, 0.04]:  # Note 0.03 is already done in main
    subprocess.run(f"expyrun configs/main.yml --noise {std} --methods {methods} --blurs {blurs}".split())

methods = "cellpose,wiener,rl,tv,tgv"
blurs = "gaussian"
for aberration in [0.08]:  # [0.02, 0.04, 0.06, 0.10]:
    subprocess.run(
        f"expyrun configs/main.yml --aberration_amplitude {aberration} --methods {methods} --blurs {blurs}".split()
    )
