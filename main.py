from config import MeasurementConfig
from pathlib import Path
import json
from sweep_gen import (
    generate_log_sweep, generate_inverse_filter, normalize_peak, pad_signal
)
from audio_io import (
    load_audio, save_audio, normalize_for_saving, check_clipping
)
from alignment import extract_aligned_segment
from deconvolution import extract_rir
from rir_processing import energy_curve, normalize_rir, trim_rir_robust
from visualization import (
    plot_rir, plot_spectrogram, plot_waveform, plot_edc, plot_together, show_all
)
from external_sweep import extract_rir, load_external_sweep, rir_from_external_sweep



def ask_user_option() -> str:
    print("\nSelect mode:")
    print("1 - Generate sweep and inverse filter")
    print("2 - Process recorded file and extract RIR")
    print("3 - Visualize all")
    print("4 - Compare 2 sine sweeps")

    choice = input("Enter your choice: ").strip()

    while choice not in {"1", "2", "3", "4"}:
        choice = input("Enter a valid choice 1/2/3/4: ").strip()
    
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

    inverse_sweep = generate_inverse_filter(
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
    rir_trimmed, trim_start, trim_end, peak_idx, envelope = trim_rir_robust(
        rir_raw,
        fs=cfg.fs,
        pre_ms=cfg.rir_pre_trim_ms,
        min_tail_ms = cfg.rir_min_tail_ms,
        threshold_over_noise_db = cfg.threshold_over_noise_db,
        arrival_smooth_ms = cfg.arrival_smooth_ms,
        tail_smooth_ms = cfg.tail_smooth_ms,
        safety_offset_ms = cfg.safety_offset_ms
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

    "direct_peak_sample": peak_idx,
    "direct_peak_seconds": peak_idx / cfg.fs,
    "rir_min_tail_ms": str(cfg.rir_min_tail_ms),
    "direct_threshold_above_noise_db": str(cfg.threshold_over_noise_db),
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
# 4. Compare 2 sine sweeps
#------------------------------------------------------------    

def visualize_together(cfg, raw_sweep):
    sweep2, fs_rec = load_audio(cfg.generated_sweep_name2, target_fs=cfg.fs, mono=True)
    plot_together(raw_sweep, sweep2, cfg.fs, "Sine sweeps together")

#------------------------------------------------------------
# 5. External Sine Sweep & Inverse Filter
#------------------------------------------------------------    
def external_sweep_rir(cfg):

    result = rir_from_external_sweep(
    sweep_path=cfg.external_sweep_path,
    recorded_path=cfg.recorded_sweep_path2,
    f_start=50.0,
    f_end=22000.0,
    target_fs=cfg.fs,
    mono=True,
)
    sweep_for_plot = result["sweep"]
    recorded = result["recorded"]
    inverse_filter = result["inverse_filter"]
    lag = result["lag"]
    rir_raw = result["rir_raw"]
    clipped = check_clipping(recorded)

    save_audio(cfg.output_dir / cfg.inverse_filter_filename, normalize_peak(inverse_filter), result["fs"])

    
    rir_trimmed, trim_start, trim_end, peak_idx, envelope = trim_rir_robust(
        rir_raw,
        fs=result["fs"],
        pre_ms=cfg.rir_trim_pre_ms,
        min_tail_ms=cfg.rir_min_tail_ms,
        threshold_above_noise_db=cfg.direct_threshold_above_noise_db,
        noise_margin_db=cfg.trim_noise_margin_db,
        arrival_smooth_ms=cfg.arrival_smooth_ms,
        tail_smooth_ms=cfg.tail_smooth_ms,
)
    


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

    elif choice =="4":
        raw_sweep, padded_sweep, inverse_filter = generate_sweep_files(cfg)
        visualize_together(cfg, raw_sweep)



if __name__ == "__main__":
    main()
