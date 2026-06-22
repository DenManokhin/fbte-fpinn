import numpy as np
from src.matrices import get_l1_weights, get_riesz_matrix_np

class FDMSolver:
    def __init__(self, params):
        self.p = params
        self.dt = (self.p["T_RANGE"][1] - self.p["T_RANGE"][0]) / (self.p["N_TIME"] - 1)
        self.dx = (self.p["X_RANGE"][1] - self.p["X_RANGE"][0]) / (self.p["N_SPACE"] - 1)
        self.x_grid = np.linspace(self.p["X_RANGE"][0], self.p["X_RANGE"][1], self.p["N_SPACE"])
        self.t_grid = np.linspace(self.p["T_RANGE"][0], self.p["T_RANGE"][1], self.p["N_TIME"])

    def solve(self):
        print(f"[FDM] Generating True Solution (Alpha={self.p['ALPHA']}, Beta={self.p['BETA']})...")
        alpha, beta = self.p["ALPHA"], self.p["BETA"]
        n_s = self.p["N_SPACE"]

        S = get_riesz_matrix_np(n_s, beta, self.dx)
        b = get_l1_weights(self.p["N_TIME"], alpha, self.dt)

        Op_diag = b[0] * np.eye(n_s) + self.p["DIFFUSION"] * S + self.p["RELAX"] * np.eye(n_s)
        Coupling = np.diag(self.p["COUPLING_CONST"] * self.x_grid)

        Top = np.hstack([Op_diag, -Coupling])
        Bot = np.hstack([Coupling, Op_diag])
        LHS = np.vstack([Top, Bot])

        # Enforce Dirichlet BCs exactly in FDM matrix
        LHS[0, :] = 0; LHS[0, 0] = 1
        LHS[n_s-1, :] = 0; LHS[n_s-1, n_s-1] = 1
        LHS[n_s, :] = 0; LHS[n_s, n_s] = 1
        LHS[-1, :] = 0; LHS[-1, -1] = 1

        u = np.exp(-10 * self.x_grid**2)
        v = np.zeros_like(self.x_grid)
        u[0] = u[-1] = 0

        history_u = np.zeros((self.p["N_TIME"], n_s))
        history_v = np.zeros((self.p["N_TIME"], n_s))
        history_u[0] = u

        for n in range(1, self.p["N_TIME"]):
            hist_u = np.zeros_like(u)
            hist_v = np.zeros_like(v)
            for k in range(1, n+1):
                w = b[k-1] - b[k] if k < n else b[n-1]
                hist_u += w * history_u[n-k]
                hist_v += w * history_v[n-k]

            rhs = np.concatenate([hist_u, hist_v])
            rhs[0] = rhs[n_s-1] = rhs[n_s] = rhs[-1] = 0 # BCs for RHS

            sol = np.linalg.solve(LHS, rhs)
            history_u[n], history_v[n] = sol[:n_s], sol[n_s:]

        return self.x_grid, self.t_grid, history_u, history_v
