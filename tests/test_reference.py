"""Checks the dense-grid reference of src/reference.py.  Run: python -m pytest tests"""
import numpy as np

from src.fdm_solver import FDMSolver
from src.physics import PARAMS, INVERSE_GRID
from src.reference import refined_params, reference_solution, rel_l2, solve_refined

P = {**PARAMS, **INVERSE_GRID}   # 16 x 25, fast


def test_refined_grid_is_nested():
    q = refined_params(P, 4, 8)
    assert q["N_SPACE"] == 61 and q["N_TIME"] == 193
    x, t, u, v = solve_refined(P, 4, 8)
    x0, t0, _, _ = FDMSolver(P).solve()
    assert np.allclose(x, x0) and np.allclose(t, t0) and u.shape == (25, 16) == v.shape


def test_refinement_one_is_the_coarse_solution():
    _, _, u, v = solve_refined(P, 1, 1)
    _, _, u0, v0 = FDMSolver(P).solve()
    assert np.array_equal(u, u0) and np.array_equal(v, v0)


def test_coarse_error_shrinks_with_refinement():
    # Against a fine reference, the error falls by about 2x per halving of the time step (order ~1)
    _, _, u_ref, _ = solve_refined(P, 4, 64)
    errs = [rel_l2(solve_refined(P, 4, r)[2], u_ref) for r in (1, 2, 4, 8)]
    assert all(a > 1.6 * b for a, b in zip(errs[:-1], errs[1:])), errs


def test_reference_is_cached(tmp_path):
    a = reference_solution(P, 2, 4, cache_dir=str(tmp_path))
    files = list(tmp_path.glob("reference_16x25_rx2_rt4_*.npz"))
    assert len(files) == 1
    b = reference_solution(P, 2, 4, cache_dir=str(tmp_path))
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    # a different equation must not reuse the cached file
    reference_solution({**P, "BETA": 1.4}, 2, 4, cache_dir=str(tmp_path))
    assert len(list(tmp_path.glob("reference_16x25_rx2_rt4_*.npz"))) == 2
