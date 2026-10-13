"""Checks that the discrete operators match eqs. (12)-(13) and that all solvers share one sign convention.

Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pytest
import torch
from scipy.special import gamma

from src.fdm_solver import FDMSolver
from src.matrices import get_l1_weights, get_riesz_matrix_np, torch_l1_matrix, torch_riesz_matrix
from src.physics import PARAMS, INVERSE_GRID, fbte_residual, interior_mask

DTYPE = torch.float64
CPU = torch.device("cpu")


@pytest.mark.parametrize("alpha", [0.3, 0.8])
def test_l1_matrix_caputo_of_t_squared(alpha):
    # Caputo derivative of t^2 is 2 t^(2-a) / Gamma(3-a); the L1 scheme converges with order 2-a
    n = 401
    t = np.linspace(0, 1, n)
    M = torch_l1_matrix(n, alpha, t[1] - t[0], CPU, dtype=DTYPE).numpy()
    exact = 2 * t ** (2 - alpha) / gamma(3 - alpha)
    assert np.allclose(M @ t ** 2, exact, atol=5e-3)
    assert np.all(M[0] == 0)


def test_l1_matrix_matches_numpy_weights():
    n, alpha, dt = 25, 0.8, 1 / 24
    M = torch_l1_matrix(n, alpha, dt, CPU, dtype=DTYPE).numpy()
    b = get_l1_weights(n, alpha, dt)
    assert np.isclose(M[1, 1], b[0]) and np.isclose(M[1, 0], -b[0])
    for i in range(2, n):
        assert np.isclose(M[i, i], b[0])
        assert np.isclose(M[i, 0], -b[i - 1])
        assert np.allclose(M[i, 1:i], (b[i - 1:0:-1] - b[i - 2::-1])[: i - 1])


def test_riesz_beta_two_is_laplacian():
    n, h = 9, 0.25
    S = get_riesz_matrix_np(n, 2.0, h)
    lap = (np.eye(n, k=-1) - 2 * np.eye(n) + np.eye(n, k=1)) / h ** 2
    assert np.allclose(S, lap, atol=1e-10)


def test_riesz_matches_fourier_symbol():
    # On R the Riesz derivative has the Fourier symbol -|k|^beta
    beta = 1.6
    x = np.linspace(-20, 20, 2001)
    h = x[1] - x[0]
    u = np.exp(-x ** 2)
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=h)
    ref = np.real(np.fft.ifft(-np.abs(k) ** beta * np.fft.fft(u)))
    approx = get_riesz_matrix_np(len(x), beta, h) @ u
    centre = slice(950, 1051)
    assert ref[1000] < 0
    assert np.max(np.abs(approx[centre] - ref[centre])) < 1e-3


def test_riesz_torch_matches_numpy():
    n, beta, length = 16, 1.6, 2.0
    S_np = get_riesz_matrix_np(n, beta, length / (n - 1))
    S_t = torch_riesz_matrix(n, beta, length, CPU, dtype=DTYPE).numpy()
    assert np.allclose(S_np, S_t, rtol=1e-12, atol=1e-10)


def test_fdm_solution_decays():
    p = {**PARAMS, **INVERSE_GRID}
    _, _, u, v = FDMSolver(p).solve()
    mag = np.sqrt(u ** 2 + v ** 2).max(axis=1)
    assert np.all(np.isfinite(mag))
    assert mag[-1] < mag[0]
    assert np.allclose(u[:, [0, -1]], 0, atol=1e-12) and np.allclose(v[:, [0, -1]], 0, atol=1e-12)


@pytest.mark.parametrize("grid", [INVERSE_GRID, {"N_SPACE": 64, "N_TIME": 100}])
def test_fdm_solution_satisfies_pinn_residual(grid):
    # The PINN residual and the FDM solver must describe the same equation,
    # so the FDM solution has (numerically) zero residual at every interior node.
    p = {**PARAMS, **grid}
    x, t, u, v = FDMSolver(p).solve()
    dt = t[1] - t[0]
    M_t = torch_l1_matrix(p["N_TIME"], p["ALPHA"], dt, CPU, dtype=DTYPE)
    M_s = torch_riesz_matrix(p["N_SPACE"], p["BETA"], x[-1] - x[0], CPU, dtype=DTYPE)
    u_t, v_t = torch.from_numpy(u), torch.from_numpy(v)
    x_t = torch.from_numpy(np.broadcast_to(x, u.shape).copy())
    res_u, res_v = fbte_residual(u_t, v_t, x_t, M_t, M_s, p)
    mask = interior_mask(u_t).bool()
    assert res_u[mask].abs().max() < 1e-8
    assert res_v[mask].abs().max() < 1e-8
