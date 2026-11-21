import numpy as np
import torch
from scipy.special import gamma, gammaln


def get_l1_weights(n, alpha, dt):
    """L1 finite difference weights for Caputo time derivative (NumPy)."""
    k = np.arange(n)
    b_k = (np.power(k + 1, 1.0 - alpha) - np.power(k, 1.0 - alpha))
    scale = 1.0 / (gamma(2.0 - alpha) * (dt ** alpha))
    return b_k * scale

def get_riesz_matrix_np(n, beta, dx):
    """Riesz fractional derivative matrix S (NumPy), eq. (13).

    S @ u approximates the Riesz derivative d^beta u / d|x|^beta, which is
    negative definite (for beta=2 it is the usual Laplacian).
    """
    idx = np.arange(n)
    i, j = np.meshgrid(idx, idx, indexing='ij')
    k = np.abs(i - j)

    x1 = beta / 2.0 - k + 1.0
    x2 = beta / 2.0 + k + 1.0

    log_num = gammaln(beta + 1.0)
    log_den = gammaln(x1) + gammaln(x2)

    sign1 = np.where(x1 < 0, (-1.0) ** (np.floor(-x1) + 1.0), 1.0)
    w_k = ((-1.0)**k) * sign1 * np.exp(log_num - log_den)
    return -w_k / (dx**beta)

def torch_riesz_matrix(n, order, domain_len, dev, dtype=None):
    """Differentiable Riesz matrix generator (PyTorch), same convention as get_riesz_matrix_np."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    order = torch.as_tensor(order, dtype=dtype, device=dev)
    h = domain_len / (n - 1)
    idx = torch.arange(n, dtype=dtype, device=dev)
    i, j = torch.meshgrid(idx, idx, indexing='ij')
    k = torch.abs(i - j)

    x1 = order / 2.0 - k + 1.0
    x2 = order / 2.0 + k + 1.0

    log_num = torch.lgamma(order + 1.0)
    log_den = torch.lgamma(x1) + torch.lgamma(x2)

    sign1 = torch.where(x1 < 0, (-1.0) ** (torch.floor(-x1) + 1.0), 1.0).to(dev)
    weights = ((-1.0) ** k) * sign1 * torch.exp(log_num - log_den)
    return -weights * (1.0 / (h ** order))

def torch_l1_matrix(n, order, dt, dev, dtype=None):
    """Differentiable L1 matrix generator for Caputo derivative (PyTorch), eq. (12)."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    order = torch.as_tensor(order, dtype=dtype, device=dev)
    k = torch.arange(n, dtype=dtype, device=dev)
    b_k = torch.pow(k + 1, 1.0 - order) - torch.pow(k, 1.0 - order)
    scale = 1.0 / (dt**order * torch.exp(torch.lgamma(2.0 - order)))

    # Row i: b_0 on the diagonal, b_m - b_{m-1} at column i-m (0 < m < i), -b_{i-1} at column 0.
    # Built from gathers instead of a Python loop so it stays cheap when rebuilt every iteration.
    idx = torch.arange(n, device=dev)
    m = idx[:, None] - idx[None, :]
    m_safe = m.clamp(min=1, max=n - 1)
    diffs = b_k[m_safe] - b_k[m_safe - 1]
    first_col = -b_k[(idx - 1).clamp(min=0)][:, None].expand(n, n)
    zero = torch.zeros((), dtype=dtype, device=dev)

    mat = torch.where(m == 0, b_k[0], zero)
    mat = torch.where((m > 0) & (idx[None, :] > 0), diffs, mat)
    mat = torch.where((idx[None, :] == 0) & (idx[:, None] > 0), first_col, mat)
    mat = torch.where(idx[:, None] == 0, zero, mat)
    return mat * scale
