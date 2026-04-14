# Code is used to find the lag between the two signals using cross-correlation, zero pad if necessary
# and return the aligned portion.

import numpy as np
from scipy.signal import correlate

# Estimate the sample delay between the source and the recorded signals.
# Positive lag -> The recording starts after the sine sweep.
# Negative lag -> The recording starts before the sine sweep (Desirable).

def estimate_delay(source: np.ndarray, recorded: np.ndarray) -> int:
    corr = correlate(recorded, source, mode= "full", method = "fft") # Here recorded and source are swapped in order. So source slides. This is done to prevent negative numbers.
    lag = np.argmax(np.abs(corr)) - (len(source) - 1) # Means the recorded signal would start 'lag' samples later.
    return int(lag)

def extract_aligned_segment(source: np.ndarray, recorded: np.ndarray) -> tuple[np.ndarray, int]:
    
    lag = estimate_delay(source, recorded)
    if lag<0: # Means the recording started after the sweep
        recorded = np.pad(recorded, (abs(lag),0))
        lag = 0

    aligned = recorded[lag:]
    return aligned, lag


    


