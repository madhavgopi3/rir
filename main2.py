import csv
from pathlib import Path
import matplotlib.pyplot as plt

from config import MeasurementConfig

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
    plot_rir, plot_spectrogram, plot_waveform, plot_edc,
    plot_deconvolution_result,
    plot_linear_and_nonlinear_ir,
    plot_linear_and_nonlinear_db,
    compute_fft_rir,
    plot_fft_rir,
    get_rir_at_freq,
    make_grid,
    plot_heatmap,
    get_octave_bands,
    show_all
)
from external_sweep import rir_from_external_sweep
from harmonic_separation import extract_ir_sweep
from acoustic_descriptors import extract_room_descriptors

def parse_points(path):
    """
    Parse points into rows and columns.
    """
    path = Path(path)
    name = path.stem.upper() # Removes path extension and converts to uppercase.
    row_label = name[0] 
    column_label = int(name[1:])
    return row_label, column_label

def save_figure(fig, output_path, dpi=150, close=True):
    """
    Saves one figure. This function is looped in save_figures
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)


def save_figures(figures, output_dir, point_name, dpi=150):
    """
    Saves all figures from a dictionary.
    """

    output_dir = Path(output_dir)

    for name, item in figures.items():
        save_path = output_dir / name / f"{point_name}_{name}.png"
        save_figure(item, save_path, dpi=dpi, close=True)

def main():
    cfg = MeasurementConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    recorded_dir = cfg.recorded_dir
    output_dir = cfg.output_dir

    output_dirs = {
        "rir_wav": output_dir / "rir_wav",
        "trimmed_rir_wav": output_dir / "trimmed_rir_wav",
        "plots": output_dir / "plots",
        "csv": output_dir / "csv",
        "heatmaps": output_dir / "heatmaps",
    }

    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    recorded_files = sorted(recorded_dir.glob("*.wav"), key=parse_points)

    if len(recorded_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {recorded_dir}")

    results = []

    # ------------------------------------------------------------
    # GENERATED SWEEP MODE
    # ------------------------------------------------------------
    if not cfg.use_external_sweep:
        raw_sweep = generate_log_sweep(
            fs=cfg.fs,
            duration=cfg.sweep_duration,
            f_start=cfg.f_start,
            f_end=cfg.f_end,
            amplitude=cfg.amplitude,
        )

        raw_sweep = normalize_peak(raw_sweep, peak=0.999)

        padded_sweep = pad_signal(
            raw_sweep,
            fs=cfg.fs,
            pre_silence=cfg.pre_silence,
            post_silence=cfg.post_silence,
        )

        inverse_filter = generate_inverse_filter(
            sweep=raw_sweep,
            fs=cfg.fs,
            f_start=cfg.f_start,
            f_end=cfg.f_end,
        )

        inverse_filter = normalize_peak(inverse_filter, peak=0.999)

        save_audio(output_dir / cfg.generated_sweep_name, raw_sweep, cfg.fs)
        save_audio(output_dir / cfg.padded_sweep_name, padded_sweep, cfg.fs)
        save_audio(output_dir / cfg.inverse_sweep_name, inverse_filter, cfg.fs)

        sweep_for_plot = raw_sweep

        reference_figures = {
            "reference_sweep": plot_waveform(
                sweep_for_plot,
                cfg.fs,
                "Reference Sweep"
            )
        }

        save_figures(
            figures=reference_figures,
            output_dir=output_dirs["plots"],
            point_name="reference",
            dpi=150,
        )

    # ------------------------------------------------------------
    # EXTERNAL SWEEP MODE
    # ------------------------------------------------------------
    
    
    else:
        sweep_for_plot, _ = load_audio(
            cfg.external_sweep_path,
            target_fs=cfg.fs,
            mono=True
        )

        reference_figures = {
            "reference_sweep": plot_waveform(
                sweep_for_plot,
                cfg.fs,
                "External Reference Sweep"
            )
        }

        save_figures(
            figures=reference_figures,
            output_dir=output_dirs["plots"],
            point_name="reference",
            dpi=150,
        )

        inverse_filter = None

    # ------------------------------------------------------------
    # PROCESS ALL 40 RECORDED FILES
    # ------------------------------------------------------------
    for rec_path in recorded_files:
        point_name = rec_path.stem.upper()
        row_label, column_label = parse_points(point_name)

        print(f"\nProcessing {point_name}")

        if not cfg.use_external_sweep:
            recorded, _ = load_audio(rec_path, target_fs=cfg.fs, mono=True)
            clipped = check_clipping(recorded)

            aligned_recording, lag = extract_aligned_segment(raw_sweep, recorded)

            rir_raw = extract_rir(aligned_recording, inverse_filter)

            ir_lin, ir_nonlin, ir_full = extract_ir_sweep(
                sweep_response=aligned_recording,
                inverse_sweep=inverse_filter,
            )

        else:
            result = rir_from_external_sweep(
                sweep_path=cfg.external_sweep_path,
                recorded_path=rec_path,
                f_start=cfg.f_start2,
                f_end=cfg.f_end2,
                target_fs=cfg.fs,
                mono=True,
            )

            sweep_for_plot = result["sweep"]
            recorded = result["recorded"]
            inverse_filter = result["inverse_filter"]
            lag = result["lag_samples"]
            rir_raw = result["rir_raw"]

            clipped = check_clipping(recorded)

            aligned_recording = result["aligned_recording"]

            ir_lin, ir_nonlin, ir_full = extract_ir_sweep(
            sweep_response=aligned_recording,
            inverse_sweep=inverse_filter,
)

        # ------------------------------------------------------------
        # COMMON POST-PROCESSING
        # ------------------------------------------------------------
        rir_trimmed, trim_start, trim_end, peak_idx, envelope = trim_rir_robust(
            rir_raw,
            fs=cfg.fs,
            pre_ms=cfg.rir_trim_pre_ms,
            min_tail_ms=cfg.rir_min_tail_ms,
            threshold_over_noise_db=cfg.threshold_over_noise_db,
            arrival_smooth_ms=cfg.arrival_smooth_ms,
            tail_smooth_ms=cfg.tail_smooth_ms,
        )

        descriptors = extract_room_descriptors(
        rir=rir_trimmed,
        fs=cfg.fs
        )

        freqs, magnitude_db = compute_fft_rir(
        h=rir_raw,
        fs=cfg.fs,
        n_fft=262144,
        )

        oct_125 = get_octave_bands(freqs, magnitude_db, 125)
        oct_250 = get_octave_bands(freqs, magnitude_db, 250)
        oct_500 = get_octave_bands(freqs, magnitude_db, 500)
        oct_1000 = get_octave_bands(freqs, magnitude_db, 1000)
        oct_2000 = get_octave_bands(freqs, magnitude_db, 2000)
        oct_4000 = get_octave_bands(freqs, magnitude_db, 4000)

        rir_trimmed_norm = normalize_rir(rir_trimmed)

        if point_name in ["C3", "C4"]:
                freq_1k, magn_1k = get_rir_at_freq(h = rir_raw, fs = cfg.fs, freq = 1000, n_fft = 262144)
                print(f"Freq at {freq_1k:2f} is {magn_1k:2f}")

        # ------------------------------------------------------------
        # SAVE AUDIO
        # ------------------------------------------------------------
        save_audio(
            output_dirs["rir_wav"] / f"{point_name}_rir_raw.wav",
            normalize_for_saving(rir_raw),
            cfg.fs,
        )

        save_audio(
            output_dirs["trimmed_rir_wav"] / f"{point_name}_rir_trimmed.wav",
            normalize_for_saving(rir_trimmed_norm),
            cfg.fs,
        )

        # ------------------------------------------------------------
        # CREATE PLOTS
        # ------------------------------------------------------------
        edc = energy_curve(rir_trimmed)

        figures = {
            "recorded_signal": plot_waveform(
                recorded,
                cfg.fs,
                f"Recorded Signal - {point_name}"
            ),
            "recorded_spectrogram": plot_spectrogram(
                recorded,
                cfg.fs,
                f"Recorded Signal Spectrogram - {point_name}"
            ),
            "rir_raw": plot_rir(
                rir_raw,
                cfg.fs,
                f"Raw Extracted RIR - {point_name}"
            ),
            "rir_trimmed": plot_rir(
                rir_trimmed_norm,
                cfg.fs,
                f"Trimmed + Normalized RIR - {point_name}"
            ),
            "frequency_response": plot_fft_rir(
                freqs,
                magnitude_db,
                f"Frequency Response from RIR - {point_name}"
            ),
            "edc": plot_edc(
                edc,
                cfg.fs,
                f"Energy Decay Curve - {point_name}"
            ),
        }

        save_figures(
            figures=figures,
            output_dir=output_dirs["plots"],
            point_name=point_name,
            dpi=150,
        )

        # ------------------------------------------------------------
        # SAVE RESULTS FOR CSV
        # ------------------------------------------------------------
        results.append({
            "point": point_name,
            "row_label": row_label,
            "column_label": column_label,
            "recording_file": rec_path.name,
            "lag_samples": lag,
            "lag_seconds": lag / cfg.fs,
            "clipped": clipped,
            "direct_peak_sample": peak_idx,
            "direct_peak_seconds": peak_idx / cfg.fs,
            "trim_start_sample": trim_start,
            "trim_end_sample": trim_end,
            "trimmed_length_samples": len(rir_trimmed_norm),
            "trimmed_length_seconds": len(rir_trimmed_norm) / cfg.fs,

            "rt20": descriptors["rt20"],
            "rt30": descriptors["rt30"],
            "c50": descriptors["c50"],
            "c80": descriptors["c80"],
            "d50": descriptors["d50"],
            "ts_ms": descriptors["ts_ms"],

            "oct_125": oct_125,
            "oct_250": oct_250,
            "oct_500": oct_500,
            "oct_1000": oct_1000,
            "oct_2000": oct_2000,
            "oct_4000": oct_4000,
        })

        print(f"Finished {point_name}")
        print(f"Lag: {lag} samples ({lag / cfg.fs:.4f} s)")
        print(f"Clipped: {clipped}")

    # ------------------------------------------------------------
    # CREATE OCTAVE BAND & DESCRIPTOR HEATMAPS
    # ------------------------------------------------------------

    octave_bands = {
        "oct_125": "125 Hz",
        "oct_250": "250 Hz",
        "oct_500": "500 Hz",
        "oct_1000": "1000 Hz",
        "oct_2000": "2000 Hz",
        "oct_4000": "4000 Hz",
    }

    for key, label in octave_bands.items():
        grid = make_grid(results, key)

        fig = plot_heatmap(
            grid,
            title=f"Octave Band Frequency Response - {label}",
            cbar_label="Magnitude [dB]",
        )

        save_figure(
            fig,
            output_dirs["heatmaps"] / f"{key}_heatmap.png",
            dpi=150,
            close=True,
        )

    heatmap_descriptors = {
        "rt20": "RT20 [s]",
        "rt30": "RT30 [s]",
        "c50": "C50 [dB]",
        "c80": "C80 [dB]",
        "d50": "D50 [-]",
        "ts_ms": "Centre Time [ms]",
    }

    for key, label in heatmap_descriptors.items():
        grid = make_grid(results, key)

        fig = plot_heatmap(
            grid,
            title=f"{key.upper()} Across Measurement Grid",
            cbar_label=label,
        )

        save_figure(
            fig,
            output_dirs["heatmaps"] / f"{key}_heatmap.png",
            dpi=150,
            close=True,
        )

    # ------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------
    csv_path = output_dirs["csv"] / "batch_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nDone.")
    print(f"Processed {len(recorded_files)} files.")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()