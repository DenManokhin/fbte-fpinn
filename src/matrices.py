import numpy as np
import torch
from scipy.special import gamma

def get_l1_weights(n, alpha, dt):
    """L1 finite difference weights for Caputo time derivative (NumPy)."""
    k = np.arange(n)
    b_k = (np.power(k + 1, 1.0 - alpha) - np.power(k, 1.0 - alpha))
    scale = 1.0 / (gamma(2.0 - alpha) * (dt ** alpha))
    return b_k * scale

def get_riesz_matrix_np(n, beta, dx):
    """Riesz fractional derivative weights (NumPy)."""
    idx = np.arange(n)
    i, j = np.meshgrid(idx, idx, indexing='ij')
    k = np.abs(i - j)
    w_k = ((-1.0)**k) * gamma(beta + 1) / (gamma(beta/2.0 - k + 1) * gamma(beta/2.0 + k + 1))
    return w_k / (dx**beta)

def torch_riesz_matrix(n, order, domain_len, dev):
    """Differentiable Riesz matrix generator (PyTorch)."""
    order = torch.as_tensor(order, dtype=torch.float32, device=dev)
    h = domain_len / (n - 1)
    idx = torch.arange(n, dtype=torch.float32, device=dev)
    i, j = torch.meshgrid(idx, idx, indexing='ij')
    k = torch.abs(i - j)

    x1 = order / 2.0 - k + 1.0
    x2 = order / 2.0 + k + 1.0

    log_num = torch.lgamma(order + 1.0)
    log_den = torch.lgamma(x1) + torch.lgamma(x2)

    sign1 = torch.where(x1 < 0, (-1.0) ** (torch.floor(-x1) + 1.0), 1.0).to(dev)
    weights = ((-1.0) ** k) * sign1 * torch.exp(log_num - log_den)
    return weights * (1.0 / (h ** order))

def torch_l1_matrix(n, order, dt, dev):
    """Differentiable L1 matrix generator for Caputo derivative (PyTorch)."""
    order = torch.as_tensor(order, dtype=torch.float32, device=dev)
    k = torch.arange(n, dtype=torch.float32, device=dev)
    b_k = torch.pow(k + 1, 1.0 - order) - torch.pow(k, 1.0 - order)
    scale = 1.0 / (dt**order * torch.exp(torch.lgamma(2.0 - order)))

    mat = torch.zeros((n, n), device=dev)
    for i in range(1, n):
        mat[i, i] = b_k[0]
        mat[i, 0] = -b_k[i-1]
        if i > 1:
            ks = torch.arange(1, i, device=dev)
            mat[i, i-ks] = b_k[ks] - b_k[ks-1]
    return mat * scale
