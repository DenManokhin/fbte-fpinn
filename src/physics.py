import torch

PARAMS = {
    "ALPHA": 0.8,         # Known Time fractional order
    "BETA": 1.6,          # Known Space fractional order
    "DIFFUSION": 0.5,     # Diffusion coefficient (D)
    "GYRO": 5.0,          # Gyromagnetic ratio
    "GRAD": 2.0,          # Gradient strength
    "RELAX": 0.5,         # R2 relaxation
    "X_RANGE": (-1, 1),
    "T_RANGE": (0, 1),
    "N_SPACE": 64,        # Grid size
    "N_TIME": 100         # Time steps
}
PARAMS["COUPLING_CONST"] = PARAMS["GYRO"] * PARAMS["GRAD"]
PARAMS["COUPLING"] = PARAMS["COUPLING_CONST"]

# The inverse problem (scripts 04 and 05) uses a coarser 16x25 grid, as stated in the paper.
INVERSE_GRID = {"N_SPACE": 16, "N_TIME": 25}


def fbte_residual(u, v, x, M_t, M_s, p, f_u=None, f_v=None):
    """PDE residuals of the discrete FBTE on a (N_TIME, N_SPACE) grid.

    Discretizes eq. (2) with Caputo/Riesz operators:
        D_t^a u =  C x v - R u + D d^b u/d|x|^b + f_u
        D_t^a v = -C x u - R v + D d^b v/d|x|^b + f_v
    where C = gamma * G, R = 1/T2, M_t is the L1 matrix and M_s the Riesz matrix (eq. 13).
    """
    u_t = torch.matmul(M_t, u)
    v_t = torch.matmul(M_t, v)
    u_riesz = torch.matmul(u, M_s.T)
    v_riesz = torch.matmul(v, M_s.T)

    c = p["COUPLING_CONST"]
    res_u = u_t - c * x * v + p["RELAX"] * u - p["DIFFUSION"] * u_riesz
    res_v = v_t + c * x * u + p["RELAX"] * v - p["DIFFUSION"] * v_riesz
    if f_u is not None:
        res_u = res_u - f_u
    if f_v is not None:
        res_v = res_v - f_v
    return res_u, res_v


def interior_mask(u):
    """Mask of collocation points where the PDE residual is enforced.

    Excludes t=0 (the L1 derivative is undefined there) and the Dirichlet nodes x=-L, x=L,
    where the solution is fixed by the boundary condition rather than by the equation.
    """
    mask = torch.ones_like(u)
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    return mask
