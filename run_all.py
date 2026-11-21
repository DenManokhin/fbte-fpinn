"""Runs all experiments of the paper and collects the tables.

Usage:
    python run_all.py                      # all paper experiments, seeds 42 1 2 3 4 for 04 and 05
    python run_all.py --only 01 02         # selected scripts
    python run_all.py --seeds 42           # a single seed for the inverse problem
    python run_all.py --skip-complexity    # skip the timing experiments (06, 07, 10)
    python run_all.py --extra              # also run 08 and 09, which are not in the paper
    python run_all.py --tables-only        # only rebuild the tables from existing results

Figures are written by the scripts into results/, logs into results/logs/, and the tables into
results/tables.md and results/table_inverse.csv. --cpu is passed on to every script.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import numpy as np

RESULTS = "results"
PAPER = ["01", "02", "03", "04", "05", "06", "07", "10"]
COMPLEXITY = ["06", "07", "10"]
EXTRA = ["08", "09"]
SEEDED = ["04", "05"]


def script_for(key):
    matches = sorted(glob.glob(f"{key}_*.py"))
    if not matches:
        raise SystemExit(f"No script for {key}")
    return matches[0]


def run(script, extra_args, log_path):
    env = {**os.environ, "MPLBACKEND": "Agg"}  # plt.show() must not block
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, script, *extra_args], stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc.returncode, time.time() - start


def read_metrics(path):
    with open(path) as f:
        rows = {r["Component"]: r for r in csv.DictReader(f)}
    u, v = rows["u (Real)"], rows["v (Imaginary)"]
    return {k: (float(u[k]), float(v[k])) for k in ("RelL2", "MSE", "MAE", "MaxAbs")}


def mean_std(values):
    values = np.asarray(values)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.4f} ± {std:.4f}"


def build_tables():
    lines = []

    for key, title in [("01_manufactured", "Table 1: fPINN vs manufactured solution (01)"),
                       ("02_forward_vs_fdm", "Table 2: fPINN vs FDM (02)")]:
        path = os.path.join(RESULTS, f"metrics_{key}.csv")
        if not os.path.exists(path):
            continue
        m = read_metrics(path)
        lines += [f"## {title}", "", "| Metric | u | v |", "|---|---|---|"]
        lines += [f"| {name} | {u:.4e} | {v:.4e} |" for name, (u, v) in m.items()]
        lines.append("")

    rows = []
    for stage in ["single", "multi"]:
        files = sorted(glob.glob(os.path.join(RESULTS, f"04_inverse_{stage}_seed*.npz")))
        if files:
            alpha = [np.load(f)["alpha"][-1] for f in files]
            beta = [np.load(f)["beta"][-1] for f in files]
            rows.append((f"no noise, {stage}-stage", len(files), alpha, beta))
    files = sorted(glob.glob(os.path.join(RESULTS, "05_inverse_noise_seed*.npz")))
    if files:
        data = [np.load(f) for f in files]
        for i, snr in enumerate(data[0]["snrs"]):
            rows.append((f"SNR {snr}", len(files), [d["alpha"][i, -1] for d in data], [d["beta"][i, -1] for d in data]))

    if rows:
        lines += ["## Inverse problem (04, 05): mean ± std over seeds", "",
                  "| Case | Seeds | alpha | beta |", "|---|---|---|---|"]
        lines += [f"| {case} | {n} | {mean_std(a)} | {mean_std(b)} |" for case, n, a, b in rows]
        lines.append("")
        with open(os.path.join(RESULTS, "table_inverse.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case", "seeds", "alpha_mean", "alpha_std", "beta_mean", "beta_std"])
            for case, n, a, b in rows:
                writer.writerow([case, n, np.mean(a), np.std(a, ddof=1) if n > 1 else 0.0,
                                 np.mean(b), np.std(b, ddof=1) if n > 1 else 0.0])

    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NN", help="scripts to run, e.g. 01 04")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4], help="seeds for 04 and 05")
    parser.add_argument("--skip-complexity", action="store_true", help="skip 06, 07 and 10")
    parser.add_argument("--extra", action="store_true", help="also run 08 and 09")
    parser.add_argument("--tables-only", action="store_true", help="only rebuild the tables")
    parser.add_argument("--cpu", action="store_true", help="run every script on the CPU")
    args = parser.parse_args()

    os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)

    if not args.tables_only:
        keys = args.only or PAPER + (EXTRA if args.extra else [])
        if args.skip_complexity:
            keys = [k for k in keys if k not in COMPLEXITY]
        common = ["--cpu"] if args.cpu else []

        jobs = []
        for key in keys:
            script = script_for(key)
            if key in SEEDED:
                jobs += [(script, ["--seed", str(s)] + common, f"{key}_seed{s}") for s in args.seeds]
            else:
                jobs.append((script, common, key))

        failed = []
        for i, (script, extra_args, name) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {script} {' '.join(extra_args)}", flush=True)
            code, seconds = run(script, extra_args, os.path.join(RESULTS, "logs", f"{name}.log"))
            print(f"    {'ok' if code == 0 else f'FAILED (exit {code})'} in {seconds:.0f} s", flush=True)
            if code != 0:
                failed.append(name)
        if failed:
            print(f"\nFailed: {', '.join(failed)}; see results/logs/")

    build_tables()


if __name__ == "__main__":
    main()
