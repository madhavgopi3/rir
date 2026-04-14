from pathlib import Path
import numpy as np

from audio_io import load_audio
from alignment import extract_aligned_segment
from deconvolution import extract_rir
from sweep_gen import generate_inverse_filter


def load_external_sweep(path, target_fs=None, mono=True):
    return load_audio(path, target_fs=target_fs, mono=mono)

def rir_from_external_sweep(
    sweep_path,
    recorded_path,
    f_start,
    f_end,
    target_fs=None,
    mono=True,
):
    raw_sweep, fs = load_external_sweep(sweep_path, target_fs=target_fs, mono=mono)
    recorded, _ = load_audio(recorded_path, target_fs=fs, mono=mono)

    inverse = generate_inverse_filter(
        sweep=raw_sweep,
        fs=fs,
        f_start=f_start,
        f_end=f_end,
    )

    aligned_recording, lag = extract_aligned_segment(raw_sweep, recorded)
    rir_raw = extract_rir(aligned_recording, inverse)

    return {
        "fs": fs,
        "sweep": raw_sweep,
        "recorded": recorded,
        "inverse_filter": inverse,
        "aligned_recording": aligned_recording,
        "lag_samples": lag,
        "lag_seconds": lag / fs,
        "rir_raw": rir_raw,
    }