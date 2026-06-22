import torch
import deepxde as dde
from src.matrices import torch_riesz_matrix, torch_l1_matrix

class ForwardGridData(dde.data.Data):
    """Custom Data class to rigidly enforce IC/BCs and PDE."""
    def __init__(self, X_grid, params, device):
        self.train_x = X_grid
        self.train_y = X_grid # Not used directly
        self.p = params
        self.device = device
        self.geom = dde.geometry.Interval(-1, 1)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        x_loc = inputs[:, 0]
        t_loc = inputs[:, 1]

        is_ic = torch.isclose(t_loc, torch.tensor(0.0, device=self.device))
        is_bc = torch.isclose(torch.abs(x_loc), torch.tensor(1.0, device=self.device))

        # Enforce IC
        u_pred_ic = outputs[is_ic, 0]
        v_pred_ic = outputs[is_ic, 1]
        u_true_ic = torch.exp(-10 * x_loc[is_ic]**2)
        v_true_ic = torch.zeros_like(u_true_ic)
        loss_ic = torch.mean((u_pred_ic - u_true_ic)**2) + torch.mean((v_pred_ic - v_true_ic)**2)

        # Enforce Dirichlet BCs
        u_pred_bc = outputs[is_bc, 0]
        v_pred_bc = outputs[is_bc, 1]
        loss_bc = torch.mean(u_pred_bc**2) + torch.mean(v_pred_bc**2)

        # Enforce FBTE
        u = outputs[:, 0:1].view(self.p["N_TIME"], self.p["N_SPACE"])
        v = outputs[:, 1:2].view(self.p["N_TIME"], self.p["N_SPACE"])
        x_grid_2d = x_loc.view(self.p["N_TIME"], self.p["N_SPACE"])

        dt = (self.p["T_RANGE"][1] - self.p["T_RANGE"][0]) / (self.p["N_TIME"] - 1)
        domain_len = self.p["X_RANGE"][1] - self.p["X_RANGE"][0]

        M_t = torch_l1_matrix(self.p["N_TIME"], self.p["ALPHA"], dt, self.device)
        M_s = torch_riesz_matrix(self.p["N_SPACE"], self.p["BETA"], domain_len, self.device)

        u_t = torch.matmul(M_t, u)
        u_xx = torch.matmul(u, M_s.T)
        v_t = torch.matmul(M_t, v)
        v_xx = torch.matmul(v, M_s.T)

        c_real = self.p["COUPLING_CONST"] * x_grid_2d * v
        c_imag = -self.p["COUPLING_CONST"] * x_grid_2d * u

        res_real = u_t + self.p["DIFFUSION"] * u_xx - c_real + self.p["RELAX"] * u
        res_imag = v_t + self.p["DIFFUSION"] * v_xx - c_imag + self.p["RELAX"] * v

        mask = torch.ones_like(u, device=self.device); mask[0, :] = 0
        loss_pde = torch.mean((res_real * mask)**2) + torch.mean((res_imag * mask)**2)

        return [loss_ic, loss_bc, loss_pde]

    def train_next_batch(self, batch_size=None): return self.train_x, self.train_y
    def test(self): return self.train_x, self.train_y

class InverseGridData(dde.data.Data):
    def __init__(self, X_train, y_train, pde_func):
        self.train_x = X_train
        self.train_y = y_train
        self.pde_func = pde_func
        self.geom = dde.geometry.Interval(-1, 1)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        loss_data = torch.mean((outputs - targets) ** 2)
        loss_physics = torch.mean(self.pde_func(inputs, outputs) ** 2)
        return [loss_data, loss_physics]

    def train_next_batch(self, batch_size=None): return self.train_x, self.train_y
    def test(self): return self.train_x, self.train_y

class MMSData(dde.data.Data):
    def __init__(self, X_train, pde_loss_fn, u_exact, v_exact, params):
        self.train_x = X_train
        self.train_y = None
        self.pde_loss_fn = pde_loss_fn
        self.u_exact = u_exact
        self.v_exact = v_exact
        self.p = params

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        pde_err = self.pde_loss_fn(inputs, outputs)
        l_pde = torch.mean(pde_err**2)

        u_net = outputs[:, 0:1].view(self.p["N_TIME"], self.p["N_SPACE"])
        v_net = outputs[:, 1:2].view(self.p["N_TIME"], self.p["N_SPACE"])

        l_ic = torch.mean((u_net[0, :] - self.u_exact[0, :])**2) + torch.mean((v_net[0, :] - self.v_exact[0, :])**2)
        l_bc = torch.mean(u_net[:, 0]**2) + torch.mean(u_net[:, -1]**2) + \
               torch.mean(v_net[:, 0]**2) + torch.mean(v_net[:, -1]**2)

        return [1 * l_pde + 100 * l_ic + 100 * l_bc]

    def train_next_batch(self, batch_size=None): return self.train_x, self.train_y
    def test(self): return self.train_x, self.train_y
