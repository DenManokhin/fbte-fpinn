"""Runs script 05 with several training settings and prints alpha, beta per SNR.

Usage:
    python sweep_noise.py --jobs 3 --seeds 42 --snrs 10 5 --config w_pde10="--w-pde 10" l2="--l2 1e-4"
Each setting writes results/05_inverse_noise_<name>_seed{N}.npz and a log in results/logs/.
Settings that already have their result file are not run again.
"""
import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--seeds", nargs="+", type=int, default=[42])
ap.add_argument("--snrs", nargs="+", default=["10", "5"])
ap.add_argument("--config", nargs="+", required=True, help='name="options for script 05"')
args = ap.parse_args()

configs = dict(c.split("=", 1) for c in args.config)
os.makedirs("results/logs", exist_ok=True)


def run(name, opts, seed):
    out = f"results/05_inverse_noise_{name}_seed{seed}.npz"
    if not os.path.exists(out):
        cmd = [sys.executable, "05_fig8_9_inverse_noise.py", *shlex.split(opts), "--tag", name,
               "--seed", str(seed), "--snrs", *args.snrs]
        with open(f"results/logs/05_{name}_seed{seed}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "MPLBACKEND": "Agg"})
    return name, seed, out


jobs = [(n, o, s) for n, o in configs.items() for s in args.seeds]
with ThreadPoolExecutor(args.jobs) as ex:
    for name, seed, out in ex.map(lambda j: run(*j), jobs):
        if not os.path.exists(out):
            print(f"{name:20s} seed {seed}: FAILED (see log)")
            continue
        d = np.load(out)
        cells = "  ".join(f"SNR {s:g}: a={a:.3f} b={b:.3f}" for s, a, b in zip(d["snrs"], d["alpha_final"], d["beta_final"]))
        print(f"{name:20s} seed {seed}: {cells}", flush=True)
