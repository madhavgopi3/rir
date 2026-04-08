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
def estimate_noise_floor(x: np.ndarray, tail_fraction: float = 0.1) -> tuple[float, float]:
    n = len(x)
    tail_samples = max(1, int(tail_fraction * n))
    tail = x[-tail_samples:]
    noise_rms = np.sqrt(np.mean(tail**2)+ 1e-12)
    noise_db = 20 * np.log10(noise_rms + 1e-12)
    return float(noise_rms), float(noise_db)

# Hilbert transform + call smoothing
# smooth_ms = Smoothing window in milliseconds
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
    
    envelope = compute_envelope(x, fs = fs, smooth_ms = smooth_ms)
    _, noise_db = estimate_noise_floor(x)
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


    
    
