from config import MeasurementConfig
from pathlib import Path
import json
from sweep_gen import (
    generate_log_sweep, generate_inverse_sweep, normalize_peak, pad_signal
)
from audio_io import (
    load_audio, save_audio, normalize_signal, check_clipping
)
from alignment import extract_aligned_segment
from deconvolution import extract_rir
from rir_processing import energy_curve, normalize_rir, trim_peak
from visualization import (
    plot_rir, plot_spectrogram, plot_waveform, plot_edc, show_all
)



def ask_user_option() -> str:
    print("\nSelect mode:")
    print("1 - Generate sweep and inverse filter only")
    print("2 - Process recorded file and extract RIR only")
    print("3 - Run full pipeline")

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

def main():
    cfg = MeasurementConfig() #cfg is a configuration object
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    choice = ask_user_option()

    if choice == "1":
        raw_sweep, padded_sweep, inverse_sweep = generate_sweep_files(cfg)
        plot_waveform(raw_sweep, cfg.fs, "Generated Raw Sweep")
        show_all()

if __name__ == "__main__":
    main()
