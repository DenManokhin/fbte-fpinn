"""Convergence study of the FDM baseline (L1 in time, Ortigueira centred difference in space).

1. Time step: N_SPACE = 64 fixed, N_TIME refined against N_TIME = 6401 on the same nodes.
2. Space step: N_TIME = 100 fixed, nested grids N_SPACE = 2^k + 1 against N_SPACE = 2049.
3. The paper grid 64 x 100 against the dense-grid reference of src/reference.py, split into the
   time and the space part, and an estimate of the error of the reference itself.
4. Independent check without any of the discrete operators: with G = 0 the problem on the whole line
   has the exact solution u_hat(k, t) = u0_hat(k) E_alpha(-(D |k|^beta + R) t^alpha), which is
   compared with the FDM solution on a wide domain [-10, 10] for |x| <= 1.

Writes results/12_fdm_convergence.md (tables) and results/12_fdm_convergence.png.
Runs on the CPU (NumPy/SciPy only), about a minute.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from src.fdm_solver import FDMSolver
from src.physics import PARAMS
from src.reference import REFINE_T, REFINE_X, reference_error_estimate, reference_solution, rel_l2, solve_refined
from src.utils import ensure_results_dir

OUT_MD = "results/12_fdm_convergence.md"
OUT_PNG = "results/12_fdm_convergence.png"


def solve(**overrides):
    return FDMSolver({**PARAMS, **overrides}).solve()


def orders(errors):
    return [""] + [f"{np.log2(a / b):.2f}" for a, b in zip(errors[:-1], errors[1:])]


def time_convergence(n_ref=6401, n_times=(26, 51, 101, 201, 401, 801, 1601)):
    _, _, u_ref, v_ref = solve(N_TIME=n_ref)
    rows = []
    for nt in n_times:
        s = (n_ref - 1) // (nt - 1)
        _, _, u, v = solve(N_TIME=nt)
        ur, vr = u_ref[::s], v_ref[::s]
        rows.append((nt, rel_l2(u, ur), rel_l2(v, vr), np.abs(u[-1] - ur[-1]).max()))
    return rows


def space_convergence(k_ref=11, ks=range(4, 10), n_time=100):
    x_r, _, u_r, v_r = solve(N_SPACE=2 ** k_ref + 1, N_TIME=n_time)
    rows = []
    for k in ks:
        s = 2 ** (k_ref - k)
        x, _, u, v = solve(N_SPACE=2 ** k + 1, N_TIME=n_time)
        ur, vr = u_r[:, ::s], v_r[:, ::s]
        inner = np.abs(x) <= 0.5
        rows.append((2 ** k + 1, rel_l2(u, ur), rel_l2(v, vr), np.abs(u[:, inner] - ur[:, inner]).max()))
    return rows


def paper_grid_error():
    _, _, u, v = FDMSolver(PARAMS).solve()
    _, _, u_ref, v_ref = reference_solution(PARAMS, REFINE_X, REFINE_T)
    _, _, u_t, v_t = solve_refined(PARAMS, 1, REFINE_T)   # converged in t, same x nodes
    _, _, u_x, v_x = solve_refined(PARAMS, REFINE_X, 1)   # converged in x, same t nodes
    est = reference_error_estimate(PARAMS, REFINE_X, REFINE_T)
    return {
        "total": (rel_l2(u, u_ref), rel_l2(v, v_ref), np.abs(u - u_ref).max(), np.abs(v - v_ref).max()),
        "time": (rel_l2(u, u_t), rel_l2(v, v_t)),
        "space": (rel_l2(u, u_x), rel_l2(v, v_x)),
        "reference": est,
    }


def mittag_leffler_neg(z, a):
    """E_a(-z) for 0 < a < 1 and z >= 0 from its integral representation."""
    if z == 0:
        return 1.0
    f = lambda r: np.exp(-r * z ** (1 / a)) * r ** (a - 1) / (r ** (2 * a) + 2 * r ** a * np.cos(a * np.pi) + 1)
    with warnings.catch_warnings():   # quad warns about roundoff once the integral is ~1e-15
        warnings.simplefilter("ignore")
        s = sum(quad(f, lo, hi, limit=200, epsabs=1e-15, epsrel=1e-12)[0] for lo, hi in [(0, 1), (1, np.inf)])
    return np.sin(a * np.pi) / np.pi * s


def analytic_check(times=(0.25, 1.0), length=10.0, n_times=(101, 401, 1601)):
    """FDM with G = 0 on [-L, L] against the exact whole-line solution, on |x| <= 1."""
    a, b, D, R = PARAMS["ALPHA"], PARAMS["BETA"], PARAMS["DIFFUSION"], PARAMS["RELAX"]
    k = np.linspace(0, 40, 1601)                   # u0_hat ~ exp(-k^2/40) is ~e^-40 at k = 40
    w = np.full(k.size, k[1] - k[0]); w[[0, -1]] /= 2
    u0_hat = np.sqrt(np.pi / 10) * np.exp(-k ** 2 / 40)
    xe = np.linspace(-1, 1, 65)
    exact = {}
    for t in times:
        g = u0_hat * np.array([mittag_leffler_neg((D * kk ** b + R) * t ** a, a) for kk in k])
        exact[t] = (np.cos(np.outer(xe, k)) * (g * w)).sum(1) / np.pi

    h = (PARAMS["X_RANGE"][1] - PARAMS["X_RANGE"][0]) / (PARAMS["N_SPACE"] - 1)   # paper h = 2/63
    n_space = int(round(2 * length / h)) + 1
    rows = []
    for nt in n_times:
        x, t, u, _ = solve(COUPLING_CONST=0.0, X_RANGE=(-length, length), N_SPACE=n_space, N_TIME=nt)
        errs = []
        for tt in times:
            j = int(np.argmin(np.abs(t - tt)))
            assert abs(t[j] - tt) < 1e-12, "the time grid must contain the comparison times"
            errs.append(np.abs(np.interp(xe, x, u[j]) - exact[tt]).max())
        rows.append((n_space, nt, *errs))
    return times, rows


def main():
    ensure_results_dir()
    p = PARAMS
    lines = [f"# FDM convergence (alpha={p['ALPHA']}, beta={p['BETA']}, D={p['DIFFUSION']}, "
             f"C={p['COUPLING_CONST']}, R={p['RELAX']})", ""]

    print("[1/4] time step")
    tr = time_convergence()
    lines += ["## Time step (N_SPACE = 64, reference N_TIME = 6401)", "",
              "| N_TIME | RelL2 u | order | RelL2 v | order | max err u at t=T | order |",
              "|---|---|---|---|---|---|---|"]
    ou, ov, oe = orders([r[1] for r in tr]), orders([r[2] for r in tr]), orders([r[3] for r in tr])
    lines += [f"| {r[0]} | {r[1]:.3e} | {ou[i]} | {r[2]:.3e} | {ov[i]} | {r[3]:.3e} | {oe[i]} |" for i, r in enumerate(tr)]

    print("[2/4] space step")
    sr = space_convergence()
    lines += ["", "## Space step (N_TIME = 100, reference N_SPACE = 2049)", "",
              "| N_SPACE | RelL2 u | order | RelL2 v | order | max err u, abs(x) <= 0.5 | order |",
              "|---|---|---|---|---|---|---|"]
    ou, ov, oe = orders([r[1] for r in sr]), orders([r[2] for r in sr]), orders([r[3] for r in sr])
    lines += [f"| {r[0]} | {r[1]:.3e} | {ou[i]} | {r[2]:.3e} | {ov[i]} | {r[3]:.3e} | {oe[i]} |" for i, r in enumerate(sr)]

    print("[3/4] paper grid vs dense reference")
    pe = paper_grid_error()
    tot, ref = pe["total"], pe["reference"]
    lines += ["", f"## Paper grid 64 x 100 vs reference ({REFINE_X}x in x, {REFINE_T}x in t)", "",
              "| | RelL2 u | RelL2 v |", "|---|---|---|",
              f"| FDM 64 x 100, total | {tot[0]:.3e} | {tot[1]:.3e} |",
              f"| time part (vs 64 x {(p['N_TIME'] - 1) * REFINE_T + 1}) | {pe['time'][0]:.3e} | {pe['time'][1]:.3e} |",
              f"| space part (vs {(p['N_SPACE'] - 1) * REFINE_X + 1} x 100) | {pe['space'][0]:.3e} | {pe['space'][1]:.3e} |",
              f"| reference itself, time (est.) | {ref['u_time']:.3e} | {ref['v_time']:.3e} |",
              f"| reference itself, space (est.) | {ref['u_space']:.3e} | {ref['v_space']:.3e} |",
              "", f"Max abs error of FDM 64 x 100: u {tot[2]:.3e}, v {tot[3]:.3e}."]

    print("[4/4] exact solution for G = 0 (takes ~30 s for the Mittag-Leffler integrals)")
    times, ar = analytic_check()
    lines += ["", "## G = 0: FDM on [-10, 10] vs exact whole-line solution, max error on abs(x) <= 1", "",
              "| N_SPACE | N_TIME | " + " | ".join(f"t = {t:g}" for t in times) + " |",
              "|---|---|" + "---|" * len(times)]
    lines += [f"| {r[0]} | {r[1]} | " + " | ".join(f"{e:.3e}" for e in r[2:]) + " |" for r in ar]

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    nt = np.array([r[0] for r in tr])
    axs[0].loglog(nt, [r[1] for r in tr], "o-", label="u")
    axs[0].loglog(nt, [r[2] for r in tr], "s-", label="v")
    axs[0].loglog(nt, tr[0][1] * (nt[0] / nt), "k--", lw=0.8, label="order 1")
    axs[0].axvline(p["N_TIME"], color="grey", ls=":", lw=1)
    axs[0].set(xlabel="N_TIME", ylabel="relative L2 error", title="Time step (N_SPACE = 64)")
    ns = np.array([r[0] for r in sr])
    axs[1].loglog(ns, [r[1] for r in sr], "o-", label="u, whole field")
    axs[1].loglog(ns, [r[3] for r in sr], "^-", label="u, max on |x| <= 0.5")
    axs[1].loglog(ns, sr[0][3] * (ns[0] / ns) ** 2, "k--", lw=0.8, label="order 2")
    axs[1].axvline(p["N_SPACE"], color="grey", ls=":", lw=1)
    axs[1].set(xlabel="N_SPACE", ylabel="error", title="Space step (N_TIME = 100)")
    for ax in axs:
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    main()
