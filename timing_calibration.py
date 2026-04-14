import numpy as np
from alignment import estimate_delay
from rir_processing import (smooth_signal, compute_envelope)

def speed_of_sound(temp_c:float = 20.0):
    return 331.3 + (0.606 * temp_c) # 343 m/s at 20 degrees.

# Setting a threshold relative to the envelope peak and returning the first meaningful index.
def first_arrival_index(h: np.ndarray,
    fs: int,
    threshold_db: float = -20.0, #Threshold relative to the envelope peak
    smooth_ms: float = 0.5,
    ) -> int:

    envelope_smooth = compute_envelope(x = h, fs = fs, smooth_ms = smooth_ms) #Uses Hilbert transform and smoothens it.
    peak = np.max(envelope_smooth)

    if(peak<1e-12):
        return 0
    
    threshold_linear = peak * (10 ** (threshold_db/20.0)) #Converting to linear scale from db and comparing it to the peak.
    # For example, if thresh_db = -20.0, the value of thresh_linear would be 0.1, signifying 10% of the peak.

    candidates = np.where(envelope_smooth >= threshold_linear)[0]

    #Handling edge cases
    if(len(candidates)==0):
        return 0
    
    return candidates[0]


    


    





