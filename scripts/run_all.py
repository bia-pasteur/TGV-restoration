# TODO: Yekta Call expyrun for running on all datasets with all parameters

import subprocess

# Run main experiment
subprocess.run("expyrun configs/main.yml")


# For abberations
methods = "cellpose,wiener,rl,tgv"
blurs = "astigmatism,coma"
for aberration in [0.1, 0.05, ...]:  # Note 0.08 is already done in main
    subprocess.run(f"expyrun configs/main.yml --aberration_ampltiude {aberration} --methods {methods} --blurs {blurs}")


# For ...
