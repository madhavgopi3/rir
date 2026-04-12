import numpy as np
from scipy.signal import hilbert

# We take the largest value with hopes that the direct arrival is the strongest peak.
def find_peak(h: np.ndarray) -> int:
    return int(np.argmax(np.abs(h)))

# Trim around RIR peak
def trim_peak(
        h: np.ndarray,
        fs: int,
        pre_ms: float = 5.0,
        post_ms: float = 1000.0
) -> tuple[np.ndarray, int, int]:
    
    peak_idx = find_peak(h)

    pre = int((pre_ms/1000.0) * fs) #Converting ms to samples
    post = int((post_ms/1000.0) * fs)

    start = max(0, peak_idx-pre)
    end = min(len(h), peak_idx + post)

    return h[start:end], start, end

def normalize_rir(h: np.ndarray, peak: float = 0.999) -> np.ndarray:
    max_val = np.max(np.abs(h))
    
    if max_val < 1e-12:
        return h.copy()
    return (h/max_val) * peak

# Returns the Schroeder's energy curve, useful or reverberation analysis like RT20, RT30, RT60.
def energy_curve(h: np.ndarray) -> np.ndarray:

    energy = h ** 2 # Energy = impulse squared.
    edc = np.cumsum(energy[::-1])[::-1] #Cumulative sum from the back. And then reverse it
    edc /= np.max(edc) + 1e-12
    return edc

def amplitude_to_db(x: np.ndarray, floor_db: float = -120.0) -> np.ndarray: 
    x_db = 20 * np.log10(np.maximum(np.abs(x), 1e-12)) #np.maximum compares two arrays element by element and keeps the larger one at each position.
    return np.maximum(x_db, floor_db)

def energy_to_db(x: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    x_db = 10.0 * np.log10(np.maximum(x, 1e-12))
    return np.maximum(x_db, floor_db)

def smooth_signal(x: np.ndarray, window_length_samples: int) -> np.ndarray:
    window_length_samples = max(1, int(window_length_samples)) # Ensures length of the window is atleast 1
    kernel = np.ones(window_length_samples, dtype=np.float64) / window_length_samples # Creates averaging kernel
    return np.convolve(x, kernel, mode="same") # Same is for making result of conv = length of x

# Function to estimate noise from the end of the RIR. tail_fraction specifies the fraction of signal for consideration
# Returns noise in rms and db
def compute_noise_floor(x: np.ndarray, tail_fraction: float = 0.1) -> tuple[float, float]:
    n = len(x)
    tail_samples = max(1, int(tail_fraction * n))
    tail = x[-tail_samples:]
    noise_rms = np.sqrt(np.mean(tail**2)+ 1e-12)
    noise_db = 20 * np.log10(noise_rms + 1e-12)
    return float(noise_rms), float(noise_db)

# Hilbert transform + call smoothing
# smooth_ms = Smoothing window in milliseconds
# We make a moving average filter. (Defined in smooth_signal()) Light filtering because smooth_ms = 1ms. Higher value, heavy smoothing.
def compute_envelope(x: np.ndarray, fs: int, smooth_ms: float = 1.0):
    hilb_sig = hilbert(x)
    envelope = np.abs(hilb_sig)
    win = max(1, int(smooth_ms/1000 * fs))
    envelope_smooth = smooth_signal(envelope, win)
    return envelope_smooth.astype(np.float64)

# Finds the meaningful arrival using smoothed envelope relative to the estimated noise floor
# Returns index of peak and the envelope
def robust_peak_finder(x: np.ndarray, 
                       fs: int,
                       threshold_over_noise_db: float = 15.0,
                       smooth_ms: float = 1.0,
                       search_start_index: int = 0) -> tuple[int, np.ndarray]:
    
    envelope = compute_envelope(x = x, fs = fs, smooth_ms = smooth_ms)
    _, noise_db = compute_noise_floor(x = x)
    env_db = 20 * np.log10(envelope + 1e-12)

    threshold_db = noise_db + threshold_over_noise_db # Detection threshold. If noise is -60 dB and TONdB is 15, all values above -45 dB will be considered.
    start_index = max(0, search_start_index)

    candidates = np.where(env_db[start_index:] > threshold_db)[0] # Returns indices of suitable candidates

    if len(candidates) == 0:
        peak_idx = int(np.argmax(envelope))
        return peak_idx, envelope
    
    start_local_idx = start_index + int(candidates[0]) # Index of the starting of the first crossing above threshold.

    # Refine a window of 2 ms after first crossing
    # 2ms because direct sound is quick. Reflections come later.
    refine_length_ms = 2.0
    refine_length_samples = max(1, int((refine_length_ms/1000.0) * fs))
    end_local_index = min(len(candidates), start_local_idx + refine_length_samples)

    local_peak_index = np.argmax(envelope[start_local_idx:end_local_index])
    peak_idx = start_index + int(local_peak_index)

    return peak_idx, envelope

# Find the end where the envelope falls back near the noise floor.
# Returns end sample index (of useful signal)

"""
EDITABLES: min_tail_ms, safety_offset_ms
SEARCH AREA: The place from the required audio peak + we add a random 300ms area. peak_idx + min_tail_samples
CANDIDATES: Indices in SEARCH AREA where the value falls below noise_db
end_idx = Place where we think the noise floor is hit + safety_offset_samples
"""

def find_noise_limited_end(
    x: np.ndarray,
    fs: int,
    peak_idx: int,
    envelope: np.ndarray | None = None,
    min_tail_ms: float = 300.0,
    smooth_ms: float = 5.0,
    safety_offset_ms = 30.0
) -> int: 
    
    if envelope == None:
        envelope = compute_envelope(x = x, fs = fs, smooth_ms = smooth_ms)
    
    _, noise_db = compute_noise_floor(x)
    envelope_db = 20 * np.log10(envelope + 1e-12)

    min_tail_samples = max(1, int((min_tail_ms/1000) * fs))
    search_start_idx = min(len(x) - 1, peak_idx + min_tail_samples) # Peak of audio signal + we add a buffer from where to start searching for the noise
    candidates = np.where(envelope_db[search_start_idx:] <= noise_db)[0] # Filter out the indices where the signal falls below noise floor.

    if len(candidates) == 0: # If there is no noise tail at the end of the signal
        return len(x) # Returns the last index of the signal itself
    
    # A small offset to ensure the tail is not cut aggressively.
    safety_offset_samples = int((safety_offset_ms/1000) * fs)
    end_idx = min(len(x), search_start_idx + candidates[0] + safety_offset_samples)

    return int(end_idx)

# This functions ties all the peak finding logic together
# Returns: trimmed_rir, start_idx, end_idx, peak_idx, envelope

def trim_rir_robust(
        x: np.ndarray,
        fs: int,
        pre_ms: float = 5.0,
        post_ms: float = 300.0,
        min_tail_ms: float = 300.0,
        threshold_over_noise_db: float = 15.0,
        arrival_smooth_ms: float = 1.0,
        tail_smooth_ms: float = 5.0,
        safety_offset_ms = 30.0
) -> tuple[np.ndarray, int, int, int, np.ndarray]:
    
    peak_idx, _ = robust_peak_finder(
        x = x, 
        fs = fs, 
        threshold_over_noise_db = threshold_over_noise_db, 
        smooth_ms = arrival_smooth_ms)
    
    pre_samples = int((pre_ms/1000) * fs)
    start_idx = max(0, peak_idx - pre_samples)

    tail_envelope = compute_envelope(x = x, fs = fs, smooth_ms= tail_smooth_ms) # Computes another envelope for tail analysis. This one is much smoother cuz of higher smooth_ms value.

    end_idx = find_noise_limited_end(
        x = x,
        fs = fs,
        peak_idx=peak_idx,
        envelope=tail_envelope,
        min_tail_ms=min_tail_ms,
        safety_offset_ms=safety_offset_ms
        )

    #Safety check if something goes wrong
    if end_idx<=start_idx:
        end = min(len(x), peak_idx+int(min_tail_ms*fs)) # keep min_tail_ms of the RIR after peak

    trimmed = x[start_idx:end_idx]
    return trimmed, start_idx, end_idx, peak_idx, tail_envelope 
""" 
We return the tail_envelope for:
decay analysis
RT60
plotting
noise floor comparison
"""

    


    





    




    
    
