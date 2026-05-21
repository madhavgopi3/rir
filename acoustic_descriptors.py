import numpy as np


def rir_energy(rir):
    rir = np.asarray(rir, dtype=np.float64).squeeze()
    return rir ** 2


def _schroeder_edc_db(rir):
    """
    Returns normalized Energy Decay Curve in dB.
    Starts near 0 dB and decays downward.
    """
    energy = rir_energy(rir)

    edc = np.cumsum(energy[::-1])[::-1]

    if np.max(edc) <= 0:
        return np.zeros_like(edc)

    edc = edc / np.max(edc)
    edc_db = 10 * np.log10(edc + 1e-12)

    return edc_db


def _linear_regression_decay_time(edc_db, fs, upper_db, lower_db):
    """
    Fits a straight line between upper_db and lower_db on the EDC.
    Used to find the slope of the decay curve.
    Example:
    RT20 uses -5 dB to -25 dB.
    RT30 uses -5 dB to -35 dB.
    EDT uses 0 dB to -10 dB.
    """
    t = np.arange(len(edc_db)) / fs

    idx = np.where((edc_db <= upper_db) & (edc_db >= lower_db))[0]

    if len(idx) < 2:
        return np.nan

    x = t[idx] # time values
    y = edc_db[idx] # EDC values in dB

    slope, intercept = np.polyfit(x, y, 1)

    if slope >= 0:
        return np.nan

    # Time needed for 60 dB decay
    rt60 = -60.0 / slope

    return rt60


def estimate_edt(rir, fs):
    """
    Early Decay Time.
    Uses 0 dB to -10 dB region, extrapolated to 60 dB.
    """
    edc_db = _schroeder_edc_db(rir)
    return _linear_regression_decay_time(edc_db, fs, 0, -10)


def estimate_rt20(rir, fs):
    """
    RT20.
    Uses -5 dB to -25 dB region, extrapolated to 60 dB.
    """
    edc_db = _schroeder_edc_db(rir)
    return _linear_regression_decay_time(edc_db, fs, -5, -25)


def estimate_rt30(rir, fs):
    """
    RT30.
    Uses -5 dB to -35 dB region, extrapolated to 60 dB.
    """
    edc_db = _schroeder_edc_db(rir)
    return _linear_regression_decay_time(edc_db, fs, -5, -35)


def clarity_c50(rir, fs):
    """
    C50 in dB.
    Ratio of early energy before 50 ms to late energy after 50 ms.
    Useful for speech clarity.
    """
    energy = rir_energy(rir)

    split = int(0.050 * fs)

    early = np.sum(energy[:split])
    late = np.sum(energy[split:])

    return 10 * np.log10((early + 1e-12) / (late + 1e-12))


def clarity_c80(rir, fs):
    """
    C80 in dB.
    Ratio of early energy before 80 ms to late energy after 80 ms.
    Useful for music clarity.
    """
    energy = rir_energy(rir)

    split = int(0.080 * fs)

    early = np.sum(energy[:split])
    late = np.sum(energy[split:])

    return 10 * np.log10((early + 1e-12) / (late + 1e-12))


def definition_d50(rir, fs):
    """
    D50.
    Fraction of total energy arriving in the first 50 ms.
    Value is between 0 and 1.
    """
    energy = rir_energy(rir)

    split = int(0.050 * fs)

    early = np.sum(energy[:split])
    total = np.sum(energy)

    return early / (total + 1e-12)


def center_time_ts(rir, fs):
    """
    Centre time Ts in milliseconds.
    Shows where the energy is concentrated in time.
    """
    energy = rir_energy(rir)
    t = np.arange(len(energy)) / fs

    ts_seconds = np.sum(t * energy) / (np.sum(energy) + 1e-12)

    return ts_seconds * 1000.0


def extract_room_descriptors(rir, fs):
    """
    Main helper function.
    Returns all broadband room descriptors in one dictionary.
    """
    rt20 = estimate_rt20(rir, fs)
    rt30 = estimate_rt30(rir, fs)
    edt = estimate_edt(rir, fs)
    c50 = clarity_c50(rir, fs)
    c80 = clarity_c80(rir, fs)
    d50 = definition_d50(rir, fs)
    ts_ms = center_time_ts(rir, fs)

    return {
        "edt": edt,
        "rt20": rt20,
        "rt30": rt30,
        "c50": c50,
        "c80": c80,
        "d50": d50,
        "ts_ms": ts_ms,
    }