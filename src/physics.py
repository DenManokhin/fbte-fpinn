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
