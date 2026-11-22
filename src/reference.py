"""Dense-grid reference solution of the forward FBTE.

The FDM solution on the paper grid (64 x 100) is not exact: its own error against the converged
solution of the same equation is about 1.7 % (u) and 2.2 % (v) in relative L2, almost all of it
from the time step (the L1 scheme converges with order ~1 here, because the solution is not smooth
at t = 0). The fPINN residual uses the same discrete operators on the same grid, so comparing the
fPINN with that FDM solution only shows how well the network solves the discrete system.

`reference_solution` solves the same scheme on a nested grid that is `refine_x` times finer in x
and `refine_t` times finer in t and returns it at the nodes of the coarse grid, so the fPINN and
the coarse FDM solution can both be measured against it. The defaults (8, 32) give a 505 x 3169
grid for the paper parameters, whose own error is ~30 times smaller than that of the 64 x 100 FDM
solution (see `reference_error_estimate` and 12_fdm_convergence.py).
"""
import hashlib
import json
import os

import numpy as np

from src.fdm_solver import FDMSolver
from src.physics import gaussian_ic
from src.utils import RESULTS_DIR

REFINE_X = 8
REFINE_T = 32

_KEYS = ("ALPHA", "BETA", "DIFFUSION", "RELAX", "COUPLING_CONST", "X_RANGE", "T_RANGE", "N_SPACE", "N_TIME")


def refined_params(params, refine_x, refine_t):
    """Parameters of the nested grid: every coarse node is also a node of the fine grid."""
    p = dict(params)
    p["N_SPACE"] = (params["N_SPACE"] - 1) * refine_x + 1
    p["N_TIME"] = (params["N_TIME"] - 1) * refine_t + 1
    return p


def solve_refined(params, refine_x, refine_t, u0=gaussian_ic):
    """Solves the FDM scheme on the refined grid and returns it at the coarse nodes."""
    x, t, u, v = FDMSolver(refined_params(params, refine_x, refine_t), u0=u0).solve()
    return x[::refine_x], t[::refine_t], u[::refine_t, ::refine_x], v[::refine_t, ::refine_x]


def _cache_path(params, refine_x, refine_t, cache_dir):
    spec = {k: params[k] for k in _KEYS}
    spec.update(refine_x=refine_x, refine_t=refine_t, ic="gaussian")
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True, default=list).encode()).hexdigest()[:10]
    name = f"reference_{params['N_SPACE']}x{params['N_TIME']}_rx{refine_x}_rt{refine_t}_{digest}.npz"
    return os.path.join(cache_dir, name)


def reference_solution(params, refine_x=REFINE_X, refine_t=REFINE_T, u0=gaussian_ic, cache_dir=RESULTS_DIR):
    """Reference solution on the coarse grid of `params`, from the FDM scheme on a nested finer grid.

    Returns x, t, u, v with u, v of shape (N_TIME, N_SPACE). With the default initial condition the
    result is cached in `cache_dir` (pass cache_dir=None to disable), keyed by the equation
    parameters and the refinement.
    """
    path = None
    if cache_dir is not None and u0 is gaussian_ic:
        path = _cache_path(params, refine_x, refine_t, cache_dir)
        if os.path.exists(path):
            d = np.load(path)
            return d["x"], d["t"], d["u"], d["v"]

    print(f"[Reference] FDM on the {refine_x}x / {refine_t}x refined grid "
          f"({(params['N_SPACE'] - 1) * refine_x + 1} x {(params['N_TIME'] - 1) * refine_t + 1})")
    x, t, u, v = solve_refined(params, refine_x, refine_t, u0)
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, x=x, t=t, u=u, v=v)
    return x, t, u, v


def rel_l2(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def reference_error_estimate(params, refine_x=REFINE_X, refine_t=REFINE_T, u0=gaussian_ic):
    """Estimates the error of the reference itself from one halving of each refinement factor.

    For a scheme of order ~1 in t and ~1-2 in x, the difference between the (rx, rt) and the
    (rx, rt/2) solutions approximates the time error of the reference (an upper bound for order > 1),
    and likewise for x. Returns relative L2 estimates {"u_time", "v_time", "u_space", "v_space"}.
    """
    _, _, u, v = solve_refined(params, refine_x, refine_t, u0)
    _, _, u_t, v_t = solve_refined(params, refine_x, refine_t // 2, u0)
    _, _, u_x, v_x = solve_refined(params, refine_x // 2, refine_t, u0)
    return {"u_time": rel_l2(u_t, u), "v_time": rel_l2(v_t, v),
            "u_space": rel_l2(u_x, u), "v_space": rel_l2(v_x, v)}
