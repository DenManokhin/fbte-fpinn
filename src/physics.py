PARAMS = {
    "ALPHA": 0.8,         # Known Time fractional order
    "BETA": 1.6,          # Known Space fractional order
    "DIFFUSION": 0.5,     # Diffusion coefficient (D)
    "GYRO": 5.0,          # Gyromagnetic ratio
    "GRAD": 2.0,          # Gradient strength
    "RELAX": 0.5,         # R2 relaxation
    "X_RANGE": (-1, 1),
    "Y_RANGE": (-1, 1),
    "T_RANGE": (0, 1),
    "N_SPACE": 64,        # Grid size (1D)
    "N_SPACE_X": 32,      # Grid size X (2D)
    "N_SPACE_Y": 32,      # Grid size Y (2D)
    "N_TIME": 100,        # Time steps (1D)
    "N_TIME_2D": 50       # Time steps (2D)
}
PARAMS["COUPLING_CONST"] = PARAMS["GYRO"] * PARAMS["GRAD"]
PARAMS["COUPLING"] = PARAMS["COUPLING_CONST"]
