import numpy as np
from scipy.signal import hilbert

EPS = 1e-30

def _as_1d_float(x):
    return np.asarray(x, dtype=np.float64).squeeze()


def _power_to_db(x):
    return 10.0 * np.log10(np.maximum(x, EPS))


def find_direct_sound_index(rir, fs, search_ms=20.0):
    """
    Finds the direct sound / main arrival using the smoothed Hilbert envelope.
    """
    h = _as_1d_float(rir)

    if len(h) == 0:
        return 0

    envelope = np.abs(hilbert(h))

    smooth_samples = max(1, int(0.001 * fs))  # 1 ms smoothing
    kernel = np.ones(smooth_samples, dtype=np.float64) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode="same")

    peak_index = int(np.argmax(envelope))

    search_samples = max(1, int((search_ms / 1000.0) * fs))
    start = max(0, peak_index - search_samples)
    noise_region = envelope[-max(1, len(envelope) // 10):]
    noise = np.sqrt(np.mean(noise_region ** 2) + EPS)

    threshold = noise * (10.0 ** (15.0 / 20.0))  # 15 dB above tail noise

    candidates = np.where(envelope[start:peak_index + 1] >= threshold)[0]

    if len(candidates) == 0:
        return peak_index

    return start + int(candidates[0])


def lundeby_knee(
    rir,
    fs,
    block_ms,
    tail_fraction,
    margin_db,
    max_iter,
):
    """
    Lundeby-style knee estimate.

    Returns:
        knee_sample: sample where decay meets the estimated noise floor
        noise_power: average noise power after the knee
        noise_db: noise floor in dB
    """
    h = _as_1d_float(rir)
    n = len(h)

    if n == 0:
        return 0, 0.0, -np.inf

    energy = h ** 2

    block_size = max(1, int((block_ms / 1000.0) * fs))
    num_blocks = n // block_size

    if num_blocks < 3:
        noise_power = float(np.mean(energy) + EPS)
        return n, noise_power, _power_to_db(noise_power)

    usable = energy[:num_blocks * block_size]
    block_power = np.mean(usable.reshape(num_blocks, block_size), axis=1)
    block_db = _power_to_db(block_power)

    tail_blocks = max(1, int(tail_fraction * num_blocks))
    noise_power = float(np.mean(block_power[-tail_blocks:]) + EPS)
    noise_db = _power_to_db(noise_power)

    peak_block = int(np.argmax(block_db))
    knee_block = num_blocks - 1

    for _ in range(max_iter):
        old_knee = knee_block

        search = block_db[peak_block:knee_block + 1]
        valid = np.where(search > noise_db + margin_db)[0]

        if len(valid) < 2:
            break

        fit_end = peak_block + int(valid[-1])

        x = np.arange(peak_block, fit_end + 1, dtype=np.float64) * (block_size / fs)
        y = block_db[peak_block:fit_end + 1]

        slope, intercept = np.polyfit(x, y, 1)

        if slope >= 0:
            break

        knee_time = (noise_db - intercept) / slope
        knee_block = int(round(knee_time * fs / block_size))
        knee_block = int(np.clip(knee_block, peak_block + 1, num_blocks - 1))

        safety_blocks = max(1, int(0.050 * fs / block_size))  # 50 ms
        noise_start = min(knee_block + safety_blocks, num_blocks - 1)

        if noise_start < num_blocks - 1:
            noise_power = float(np.mean(block_power[noise_start:]) + EPS)
            noise_db = _power_to_db(noise_power)

        if abs(knee_block - old_knee) <= 1:
            break

    knee_sample = min(n, knee_block * block_size)

    if knee_sample < n - 1:
        noise_power = float(np.mean(energy[knee_sample:]) + EPS)
        noise_db = _power_to_db(noise_power)

    return knee_sample, noise_power, noise_db


def schroeder_edc_db(
    rir,
    fs,
    noise_compensate,
    block_ms,
    tail_fraction,
    margin_db,
    max_iter,
):
    """
    Returns a normalized Schroeder Energy Decay Curve in dB.

    If noise_compensate=True:
    - finds a Lundeby-style knee and integrates only to the knee
    """
    h = _as_1d_float(rir)
    n = len(h)

    if n == 0:
        return np.array([], dtype=np.float64), 0, 0.0, -np.inf

    energy = h ** 2

    if not noise_compensate:
        edc = np.cumsum(energy[::-1])[::-1]
        max_edc = np.max(edc)
        if max_edc <= 0:
            return np.zeros_like(h), n, 0.0, -np.inf
        edc = edc / max_edc
        return _power_to_db(edc), n, 0.0, -np.inf

    knee, noise_power, noise_db = lundeby_knee(
    h,
    fs,
    block_ms=block_ms,
    tail_fraction=tail_fraction,
    margin_db=margin_db,
    max_iter=max_iter,
)

    knee = int(np.clip(knee, 1, n))
    useful_energy = energy[:knee]

    raw_edc = np.cumsum(useful_energy[::-1])[::-1]

    # Expected accumulated noise energy from each time sample to the knee.
    remaining_samples = np.arange(knee, 0, -1, dtype=np.float64)
    noise_correction = noise_power * remaining_samples

    edc = raw_edc - noise_correction
    edc[edc < EPS] = EPS

    full_edc = np.full(n, EPS, dtype=np.float64)
    full_edc[:knee] = edc

    full_edc = full_edc / np.max(full_edc)
    edc_db = _power_to_db(full_edc)

    return edc_db, knee, noise_power, noise_db


def decay_time_from_edc(
    edc_db,
    fs,
    upper_db,
    lower_db,
    min_points=50,
    min_r2=0.90,
    min_rt=0.03,
    max_rt=3.0,
):
    edc_db = _as_1d_float(edc_db)

    t = np.arange(len(edc_db), dtype=np.float64) / fs

    idx = np.where((edc_db <= upper_db) & (edc_db >= lower_db))[0]

    if len(idx) < min_points:
        return np.nan

    x = t[idx]
    y = edc_db[idx]

    slope, intercept = np.polyfit(x, y, 1)

    if not np.isfinite(slope) or slope >= 0:
        return np.nan

    y_fit = slope * x + intercept

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot <= 0:
        return np.nan

    r2 = 1.0 - ss_res / ss_tot

    rt60 = -60.0 / slope

    return rt60


def clarity_c50(rir, fs, direct_index=0):
    h = _as_1d_float(rir)
    energy = h ** 2

    start = int(np.clip(direct_index, 0, len(h)))
    split = min(len(h), start + int(0.050 * fs))

    early = np.sum(energy[start:split])
    late = np.sum(energy[split:])

    return 10.0 * np.log10((early + EPS) / (late + EPS))


def clarity_c80(rir, fs, direct_index=0):
    h = _as_1d_float(rir)
    energy = h ** 2

    start = int(np.clip(direct_index, 0, len(h)))
    split = min(len(h), start + int(0.080 * fs))

    early = np.sum(energy[start:split])
    late = np.sum(energy[split:])

    return 10.0 * np.log10((early + EPS) / (late + EPS))


def definition_d50(rir, fs, direct_index=0):
    h = _as_1d_float(rir)
    energy = h ** 2

    start = int(np.clip(direct_index, 0, len(h)))
    split = min(len(h), start + int(0.050 * fs))

    early = np.sum(energy[start:split])
    total = np.sum(energy[start:])

    return early / (total + EPS)


def center_time_ts(rir, fs, direct_index=0):
    h = _as_1d_float(rir)
    energy = h ** 2

    start = int(np.clip(direct_index, 0, len(h)))
    useful_energy = energy[start:]

    t = np.arange(len(useful_energy), dtype=np.float64) / fs
    ts = np.sum(t * useful_energy) / (np.sum(useful_energy) + EPS)

    return ts * 1000.0


def extract_room_descriptors(
    rir,
    fs,
    noise_compensate,
    direct_sound_search_ms,
    lundeby_block_ms,
    lundeby_tail_fraction,
    lundeby_margin_db,
    lundeby_max_iter,
    direct_index=None,
):
    """
    Calculates broadband room-acoustic descriptors from one RIR.

    Returns:
        edt, rt20, rt30 in seconds
        c50, c80 in dB
        d50 as a ratio from 0 to 1
        ts_ms in milliseconds
    """
    h = _as_1d_float(rir)

    if direct_index is None:
        direct_index = find_direct_sound_index(h, fs, search_ms=direct_sound_search_ms,
        )

        # If the detected direct sound is extremely close to the start,
        # treat the RIR as already trimmed to direct sound.
        if direct_index < int(0.002 * fs):
            direct_index = 0

    edc_db, knee, noise_power, noise_db = schroeder_edc_db(
    h,
    fs,
    noise_compensate=noise_compensate,
    block_ms=lundeby_block_ms,
    tail_fraction=lundeby_tail_fraction,
    margin_db=lundeby_margin_db,
    max_iter=lundeby_max_iter,
)

    edt = decay_time_from_edc(edc_db, fs, -1.0, -10.0)
    rt20 = decay_time_from_edc(edc_db, fs, -5.0, -25.0)
    rt30 = decay_time_from_edc(edc_db, fs, -5.0, -35.0)

    return {
        "edt": edt,
        "rt20": rt20,
        "rt30": rt30,
        "c50": clarity_c50(h, fs, direct_index),
        "c80": clarity_c80(h, fs, direct_index),
        "d50": definition_d50(h, fs, direct_index),
        "ts_ms": center_time_ts(h, fs, direct_index),

        "direct_index": direct_index,
        "direct_time_s": direct_index / fs,
        "lundeby_knee_sample": knee,
        "lundeby_knee_s": knee / fs,
        "noise_power": noise_power,
        "noise_db": noise_db,
    }




