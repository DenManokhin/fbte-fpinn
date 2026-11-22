"""Baseline for the inverse problem: FDM solver + nonlinear least squares.

The fractional orders are recovered by fitting the implicit FDM solution to the data with
scipy.optimize.least_squares, on the same 16x25 grid, initial guess and bounds as the fPINN
(alpha in (0, 1), beta in (1, 2), start at alpha = 0.5, beta = 1.5).

Two kinds of data are fitted:
  * the exact data the fPINN was trained on, taken from results/05_inverse_noise_seed{N}.npz
    when script 05 has been run (so the comparison with Fig. 13 is like for like);
  * N_REAL independent noise realisations per SNR (--realisations N, default 50), which give
    the mean and spread of the estimates.

--fpinn TAG compares with a non-default fPINN setting of script 05, i.e. the files
results/05_inverse_noise_TAG_seed{N}.npz (default: the paper's setting). The same-data comparison,
mean ± std over the seeds, is written to results/11_same_data[_TAG].md.

Like the fPINN, the fit uses the same discretisation that generated the data, so neither
method is affected by a model mismatch here.
"""
import glob
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.optimize import least_squares

from src.physics import PARAMS, INVERSE_GRID
from src.fdm_solver import FDMSolver
from src.utils import add_complex_gaussian_noise, ensure_results_dir

ensure_results_dir()
P = {**PARAMS, **INVERSE_GRID}
SNRS = [np.inf, 50, 20, 10, 5]
N_REAL = int(sys.argv[sys.argv.index("--realisations") + 1]) if "--realisations" in sys.argv else 50
FPINN_TAG = sys.argv[sys.argv.index("--fpinn") + 1] if "--fpinn" in sys.argv else ""
FPINN_SUFFIX = f"_{FPINN_TAG}" if FPINN_TAG else ""
X0 = np.array([0.5, 1.5])
BOUNDS = ([1e-3, 1.0 + 1e-3], [1.0 - 1e-3, 2.0 - 1e-3])


def label(snr):
    return f"SNR={snr:g}" if np.isfinite(snr) else "clean data"


def simulate(theta):
    _, _, u, v = FDMSolver({**P, "ALPHA": theta[0], "BETA": theta[1]}).solve()
    return u, v


def fit(u_data, v_data):
    """Least-squares estimate of (alpha, beta) from data on the 16x25 grid."""
    def residual(theta):
        u, v = simulate(theta)
        return np.concatenate([(u - u_data).ravel(), (v - v_data).ravel()])
    t0 = time.perf_counter()
    sol = least_squares(residual, X0, bounds=BOUNDS, x_scale=[0.1, 0.1])
    return sol.x, time.perf_counter() - t0, sol.nfev


def quiet_fdm():
    # FDMSolver.solve prints a line on every call; the fit calls it hundreds of times
    import builtins
    original = builtins.print
    builtins.print = lambda *a, **k: None if (a and str(a[0]).startswith("[FDM]")) else original(*a, **k)


def write_same_data_table(same_data):
    """fPINN and FDM-LS on the same noisy data: mean ± std over the seeds of script 05."""
    def ms(z):
        z = np.asarray(z)
        return f"{z.mean():.3f} ± {z.std(ddof=1) if len(z) > 1 else 0.0:.3f}"
    snrs = sorted({snr for _, snr in same_data}, key=lambda s: -s)
    lines = [f"fPINN setting: {FPINN_TAG or 'paper'}; true alpha = {P['ALPHA']}, beta = {P['BETA']}", "",
             "| Case | Seeds | fPINN alpha | fPINN beta | FDM-LS alpha | FDM-LS beta |", "|---|---|---|---|---|---|"]
    for snr in snrs:
        rows = np.array([v for (_, s), v in same_data.items() if s == snr])
        lines.append(f"| {label(snr)} | {len(rows)} | {ms(rows[:, 0])} | {ms(rows[:, 1])} | "
                     f"{ms(rows[:, 2])} | {ms(rows[:, 3])} |")
    path = f"results/11_same_data{FPINN_SUFFIX}.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n(written to {path})")


def run_experiment():
    quiet_fdm()
    u_true, v_true = simulate([P["ALPHA"], P["BETA"]])
    print(f"--- FDM + least squares baseline (grid {P['N_SPACE']}x{P['N_TIME']}, "
          f"{N_REAL} noise realisations per SNR) ---")

    # 1. The exact data the fPINN saw, when script 05 has been run
    same_data = {}
    for path in sorted(glob.glob(f"results/05_inverse_noise{FPINN_SUFFIX}_seed*.npz")):
        d = np.load(path)
        if "u_data" not in d:
            print(f"{path} has no saved data (run the current script 05 again); skipped")
            continue
        seed = path.split("seed")[-1].split(".")[0]
        print(f"\nSame data as fPINN, {path}:")
        print("  case          fPINN alpha  fPINN beta   FDM-LS alpha  FDM-LS beta   fit time")
        # the reported fPINN estimates (early stopping, if used); older files only have the histories
        a_f = d["alpha_final"] if "alpha_final" in d else d["alpha"][:, -1]
        b_f = d["beta_final"] if "beta_final" in d else d["beta"][:, -1]
        for i, snr in enumerate(d["snrs"]):
            theta, secs, _ = fit(d["u_data"][i], d["v_data"][i])
            same_data[(seed, float(snr))] = (a_f[i], b_f[i], *theta)
            print(f"  {label(snr):12s}  {a_f[i]:.4f}       {b_f[i]:.4f}       "
                  f"{theta[0]:.4f}        {theta[1]:.4f}        {secs:.2f} s")

    if same_data:
        write_same_data_table(same_data)

    # 2. Many noise realisations per SNR
    rng_seeds = range(N_REAL)
    est = {snr: np.zeros((N_REAL if np.isfinite(snr) else 1, 2)) for snr in SNRS}
    times = []
    for snr in SNRS:
        for r in (rng_seeds if np.isfinite(snr) else [0]):
            np.random.seed(1000 + r)
            if np.isfinite(snr):
                u_d, v_d = add_complex_gaussian_noise(u_true, v_true, snr)
            else:
                u_d, v_d = u_true, v_true
            theta, secs, _ = fit(u_d, v_d)
            est[snr][r] = theta
            times.append(secs)

    print(f"\nFDM + least squares over {N_REAL} noise realisations (true alpha={P['ALPHA']}, beta={P['BETA']}):")
    print("  case          alpha mean ± std      beta mean ± std       alpha [min, max]     beta [min, max]")
    for snr in SNRS:
        a, b = est[snr][:, 0], est[snr][:, 1]
        sd = (lambda z: np.std(z, ddof=1)) if len(a) > 1 else (lambda z: 0.0)
        print(f"  {label(snr):12s}  {a.mean():.4f} ± {sd(a):.4f}     {b.mean():.4f} ± {sd(b):.4f}     "
              f"[{a.min():.4f}, {a.max():.4f}]   [{b.min():.4f}, {b.max():.4f}]")
    print(f"\nMedian fit time: {np.median(times):.2f} s (max {np.max(times):.2f} s)")

    np.savez(f"results/11_inverse_fdm_baseline{FPINN_SUFFIX}.npz", snrs=np.array(SNRS),
             alpha=np.array([est[s][:, 0] if len(est[s]) == N_REAL else np.repeat(est[s][0, 0], N_REAL) for s in SNRS]),
             beta=np.array([est[s][:, 1] if len(est[s]) == N_REAL else np.repeat(est[s][0, 1], N_REAL) for s in SNRS]),
             fit_times=np.array(times))

    # Figure: estimates vs SNR, FDM-LS mean ± std, fPINN on the same data where available
    noisy = [s for s in SNRS if np.isfinite(s)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, j, name, true in [(axs[0], 0, "alpha", P["ALPHA"]), (axs[1], 1, "beta", P["BETA"])]:
        m = [est[s][:, j].mean() for s in noisy]
        e = [est[s][:, j].std(ddof=1) for s in noisy]
        ax.errorbar(noisy, m, yerr=e, marker="o", capsize=4, label=f"FDM + least squares (mean ± std, {N_REAL} realisations)")
        pts = [(snr, vals) for (_, snr), vals in same_data.items() if np.isfinite(snr)]
        if pts:
            ax.plot([s for s, _ in pts], [v[j] for _, v in pts], "s", mfc="none", ms=9,
                    label=f"fPINN (script 05{', ' + FPINN_TAG if FPINN_TAG else ''})")
            ax.plot([s for s, _ in pts], [v[2 + j] for _, v in pts], "x", ms=9, label="FDM + least squares, same data as fPINN")
        ax.axhline(true, color="k", ls="--", label=f"true {name}")
        ax.set_xscale("log")
        ax.set_xticks(noisy, [str(s) for s in noisy])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.invert_xaxis()
        ax.set_xlabel("SNR")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.4)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/11_inverse_fdm_baseline{FPINN_SUFFIX}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_experiment()
