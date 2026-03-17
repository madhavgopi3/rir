import numpy as np

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