# Does FFT convolution to extract the RIR
# Deconvolve and RIR extraction kept as separate functions in case more features need to be added to RIR extraction.

import numpy as np
from scipy.signal import fftconvolve

def deconvolve(recorded: np.ndarray, inverse_filter: np.ndarray) -> np.ndarray:
    h = fftconvolve(recorded, inverse_filter, mode = "full")
    return h.astype(np.float64)

def extract_rir(recorded: np.ndarray, inverse_filter: np.ndarray) -> np.ndarray:
    return deconvolve(recorded, inverse_filter)