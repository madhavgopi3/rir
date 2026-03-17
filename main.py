from config import MeasurementConfig
from pathlib import Path
import json
from sweep_gen import (
    generate_log_sweep, generate_inverse_sweep, normalize_peak, pad_signal
)
from audio_io import (
    load_audio, save_audio, normalize_for_saving, check_clipping
)
from alignment import extract_aligned_segment
from deconvolution import extract_rir
from rir_processing import energy_curve, normalize_rir, trim_peak
from visualization import (
    plot_rir, plot_spectrogram, plot_waveform, plot_edc, show_all
)



def ask_user_option() -> str:
    print("\nSelect mode:")
    print("1 - Generate sweep and inverse filter")
    print("2 - Process recorded file and extract RIR")
    print("3 - Visualize all")

    choice = input("Enter your choice: ").strip()

    while choice not in {"1", "2", "3"}:
        choice = input("Enter a valid choice 1/2/3: ").strip()
    
    return choice


#------------------------------------------------------------
# 1. Generate sweep and inverse filter
#------------------------------------------------------------

def generate_sweep_files(cfg: MeasurementConfig):

    raw_sweep = generate_log_sweep(
        fs=cfg.fs,
        duration=cfg.sweep_duration,
        f_start=cfg.f_start,
        f_end=cfg.f_end,
        amplitude=cfg.amplitude
    )

    raw_sweep = normalize_peak(raw_sweep, peak=0.999)

    padded_sweep = pad_signal(
        signal=raw_sweep,
        fs=cfg.fs,
        pre_silence=cfg.pre_silence,
        post_silence=cfg.post_silence
        )

    inverse_sweep = generate_inverse_sweep(
        sweep=raw_sweep,
        fs=cfg.fs,
        duration=cfg.sweep_duration,
        f_start=cfg.f_start,
        f_end=cfg.f_end
    )

    inverse_sweep = normalize_peak(signal=inverse_sweep,peak=0.999)

    save_audio(cfg.output_dir/cfg.generated_sweep_name, raw_sweep, cfg.fs)
    save_audio(cfg.output_dir/cfg.padded_sweep_name, padded_sweep, cfg.fs)
    save_audio(cfg.output_dir/cfg.inverse_sweep_name, inverse_sweep, cfg.fs)

    return raw_sweep, padded_sweep, inverse_sweep

#------------------------------------------------------------
# 2. Processing of the recording and RIR
#------------------------------------------------------------

def process_recording(cfg: MeasurementConfig, padded_sweep, inverse_filter):
    recorded, fs_rec = load_audio(cfg.recorded_sweep_path, target_fs=cfg.fs, mono=True)
    clipped=check_clipping(recorded)

    aligned_rec, lag = extract_aligned_segment(padded_sweep, recorded)

    # Removing Pre-Silence
    pre_silence_samples = int(cfg.pre_silence * cfg.fs)
    if len(aligned_rec) > pre_silence_samples:
        aligned_active = aligned_rec[pre_silence_samples:]
    aligned_active = aligned_rec

    # Extract RIR
    rir_raw = extract_rir(aligned_active, inverse_filter)
    rir_trimmed, trim_start, trim_end = trim_peak(
        rir_raw,
        fs=cfg.fs,
        pre_ms=cfg.rir_pre_trim_ms,
        post_ms=cfg.rir_post_trim_ms
        )
    
    rir_trimmed_norm = normalize_rir(rir_trimmed)

    save_audio(cfg.output_dir/cfg.rir_name, normalize_for_saving(rir_raw), cfg.fs)
    save_audio(cfg.output_dir/cfg.trimmed_rir_name, normalize_for_saving(rir_trimmed_norm), cfg.fs)

    metadata = {
    "fs": cfg.fs,
    "sweep_duration": cfg.sweep_duration,
    "f_start": cfg.f_start,
    "f_end": cfg.f_end,
    "amplitude": cfg.amplitude,
    "pre_silence": cfg.pre_silence,
    "post_silence": cfg.post_silence,
    "recording_path": str(cfg.recorded_sweep_path),
    "output_dir": str(cfg.output_dir),
    "estimated_lag_samples": int(lag),
    "estimated_lag_seconds": float(lag / cfg.fs),
    "recording_clipped": bool(clipped),
    "trim_start_sample": int(trim_start),
    "trim_end_sample": int(trim_end),
    "trimmed_rir_length_samples": int(len(rir_trimmed_norm)),
    "trimmed_rir_length_seconds": float(len(rir_trimmed_norm) / cfg.fs),
}

    with open(cfg.output_dir/"metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent = 2)

    return recorded, clipped, lag, rir_raw, rir_trimmed_norm

#------------------------------------------------------------
# 3. Visualization
#------------------------------------------------------------

def visualize_full(cfg, raw_sweep, recorded, rir_raw, rir_trimmed_norm):
    plot_waveform(raw_sweep, cfg.fs, "Generated Log Sweep")
    plot_waveform(recorded, cfg.fs, "Uploaded Recorded Signal")
    plot_spectrogram(recorded, cfg.fs, "Recorded Signal Spectrogram")
    plot_rir(rir_raw, cfg.fs, "Raw Extracted RIR")
    plot_rir(rir_trimmed_norm, cfg.fs, "Trimmed + Normalized RIR")

    edc = energy_curve(rir_trimmed_norm)
    plot_edc(edc, cfg.fs, "Energy Decay Curve")
    show_all()

#------------------------------------------------------------
# MAIN DRIVER
#------------------------------------------------------------

def main():
    cfg = MeasurementConfig() #cfg is a configuration object
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    choice = ask_user_option()

    if choice == "1":
        raw_sweep, padded_sweep, inverse_sweep = generate_sweep_files(cfg)
        plot_waveform(raw_sweep, cfg.fs, "Generated Raw Sweep")
        show_all()

    elif choice == "2":
        raw_sweep, padded_sweep, inverse_sweep = generate_sweep_files(cfg)
        recorded, clipped, lag, rir_raw, rir_trimmed_norm = process_recording(
            cfg, padded_sweep, inverse_sweep
        )

        plot_waveform(recorded, cfg.fs, "Uploaded Recorded Signal")
        plot_rir(rir_raw, cfg.fs, "Raw Extracted RIR")
        plot_rir(rir_trimmed_norm, cfg.fs, "Trimmed + Normalized RIR")
        show_all() 

    elif choice =="3":
        raw_sweep, padded_sweep, inverse_filter = generate_sweep_files(cfg)
        recorded, clipped, lag, rir_raw, rir_trimmed_norm = process_recording(
            cfg, padded_sweep, inverse_filter
        )

        visualize_full(cfg, raw_sweep, recorded, rir_raw, rir_trimmed_norm)

if __name__ == "__main__":
    main()
