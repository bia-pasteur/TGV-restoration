import subprocess

# Main table with default parameters
subprocess.run("expyrun configs/main.yml".split())


# See the impact of different parameters

# Blur amplitude
methods = "cellpose,wiener,rl,rltv,tgv4"
datasets = "Brightfield,Fluorescence,Phase_Contrast"
for aberration in [0.02, 0.04, 0.06, 0.10]:  # Note 0.08 is already done in main
    subprocess.run(
        f"expyrun configs/main.yml --aberration_amplitude {aberration} --methods {methods} --datasets {datasets}".split()
    )


# PSF imperfection
for psf_imperfection in [0.9, 0.8, 0.7, 0.6]:  # Note 1.0 is already done in main
    subprocess.run(
        f"expyrun configs/main.yml --psf_imperfection {psf_imperfection} --methods {methods} --datasets {datasets}".split()
    )

# Noise
for std in [0, 0.01, 0.02, 0.04]:  # Note 0.03 is already done in main
    subprocess.run(f"expyrun configs/main.yml --noise {std} --methods {methods} --datasets {datasets}".split())


# Sparsity
methods = "tv,tgv,tgv3,tgv4"
for sparsity in [0, 0.0005, 0.001, 0.01]:  # Note 0.005 is already done in main
    subprocess.run(f"expyrun configs/main.yml --sparsity {sparsity} --methods {methods} --datasets {datasets}".split())
