import os
import sys
import numpy as np
import torch

def add_gaussian_noise(data, snr):
    """Adds Gaussian noise to the data based on desired SNR."""
    signal_power = np.mean(data ** 2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def select_device():
    """Selects the compute device. Reverts PyTorch to CPU if CPU execution is forced."""
    if "--cpu" in sys.argv or os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def configure_precision(device):
    """Configures global precision for PyTorch and DeepXDE based on the device."""
    import deepxde as dde
    if device.type == "mps":
        # Apple Silicon MPS does not support float64
        dde.config.set_default_float("float32")
        torch.set_default_dtype(torch.float32)
        print("MPS detected. Precision set to float32.")
    else:
        # Default to float64 for better numerical stability in finite difference matrices
        dde.config.set_default_float("float64")
        torch.set_default_dtype(torch.float64)
        print("Precision set to float64.")

def set_reproducible(seed=42):
    """Sets all random seeds for reproducibility (Python, NumPy, PyTorch, DeepXDE)."""
    import random
    import numpy as np
    import torch
    import deepxde as dde

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    dde.config.set_random_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(u_true, u_pred):
    """
    Calculates standard evaluation metrics for PDE solvers and PINNs.

    Args:
        u_true: Ground truth / analytical solution array
        u_pred: Predicted solution array

    Returns:
        Dictionary containing RelL2, MSE, MAE, MaxAbs, and PSNR.
    """
    u_true = np.asarray(u_true)
    u_pred = np.asarray(u_pred)

    metrics = {}

    # 1. Relative L2 Error
    norm_true = np.linalg.norm(u_true)
    if norm_true > 1e-12:
        metrics["RelL2"] = np.linalg.norm(u_pred - u_true) / norm_true
    else:
        metrics["RelL2"] = np.nan # Undefined if true solution is mostly zero

    # 2. Mean Squared Error (MSE)
    mse = np.mean((u_pred - u_true)**2)
    metrics["MSE"] = mse

    # 3. Mean Absolute Error (MAE / L1)
    metrics["MAE"] = np.mean(np.abs(u_pred - u_true))

    # 4. Maximum Absolute Error (L_inf)
    metrics["MaxAbs"] = np.max(np.abs(u_pred - u_true))

    # 5. Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(np.abs(u_true))
    if mse > 1e-12 and max_val > 1e-12:
        metrics["PSNR"] = 10 * np.log10((max_val**2) / mse)
    else:
        metrics["PSNR"] = float('inf') if max_val > 1e-12 else np.nan

    return metrics

def save_metrics_to_csv(u_metrics, v_metrics, filename):
    import csv
    import os
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Component", "RelL2", "MSE", "MAE", "MaxAbs", "PSNR"])
        writer.writerow(["u (Real)", u_metrics.get("RelL2"), u_metrics.get("MSE"), u_metrics.get("MAE"), u_metrics.get("MaxAbs"), u_metrics.get("PSNR")])
        writer.writerow(["v (Imaginary)", v_metrics.get("RelL2"), v_metrics.get("MSE"), v_metrics.get("MAE"), v_metrics.get("MaxAbs"), v_metrics.get("PSNR")])


