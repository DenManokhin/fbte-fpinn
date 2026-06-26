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


