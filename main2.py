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
    plot_fft_rir,
    show_all
)
from external_sweep import rir_from_external_sweep

from harmonic_separation import extract_ir_sweep
from rir_processing import normalize_rir, trim_rir_robust



def main():
    cfg = MeasurementConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.use_external_sweep:
        # ------------------------------------------------------------
        # GENERATED SWEEP MODE
        # ------------------------------------------------------------
        raw_sweep = generate_log_sweep(
            fs=cfg.fs,
            duration=cfg.sweep_duration,
            f_start=cfg.f_start,
            f_end=cfg.f_end,
            amplitude=cfg.amplitude,
        )

        raw_sweep = normalize_peak(raw_sweep, peak=0.999)

        #padded sweep is not used.
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

        save_audio(cfg.output_dir / cfg.generated_sweep_name, raw_sweep, cfg.fs)
        save_audio(cfg.output_dir / cfg.padded_sweep_name, padded_sweep, cfg.fs)
        save_audio(cfg.output_dir / cfg.inverse_sweep_name, inverse_filter, cfg.fs)

        recorded, _ = load_audio(cfg.recorded_sweep_path, target_fs=cfg.fs, mono=True)
        clipped = check_clipping(recorded)

        aligned_recording, lag = extract_aligned_segment(raw_sweep, recorded)

        rir_raw = extract_rir(aligned_recording, inverse_filter)
        sweep_for_plot = raw_sweep

        ir_lin, ir_nonlin, ir_full = extract_ir_sweep(
        sweep_response=aligned_recording,
        inverse_sweep=inverse_filter,
        )

        #plot_deconvolution_result(ir_full=ir_full, fs=cfg.fs)
        #plot_linear_and_nonlinear_ir(ir_lin=ir_lin, ir_nonlin=ir_nonlin, fs=cfg.fs)
        #plot_linear_and_nonlinear_db(ir_lin=ir_lin, ir_nonlin=ir_nonlin, fs=cfg.fs)

    else:
        # ------------------------------------------------------------
        # EXTERNAL SWEEP MODE
        # ------------------------------------------------------------
        result = rir_from_external_sweep(
            sweep_path=cfg.external_sweep_path,
            recorded_path=cfg.recorded_sweep_path2,
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

        save_audio(cfg.output_dir / cfg.external_inverse_name, normalize_peak(inverse_filter), cfg.fs)
        
        ir_lin, ir_nonlin, ir_full = extract_ir_sweep(
        sweep_response=recorded,
        inverse_sweep=inverse_filter,
        )

       

        #plot_deconvolution_result(ir_full=ir_full, fs=cfg.fs)
        #plot_linear_and_nonlinear_ir(ir_lin=ir_lin, ir_nonlin=ir_nonlin, fs=cfg.fs)
        #plot_linear_and_nonlinear_db(ir_lin=ir_lin, ir_nonlin=ir_nonlin, fs=cfg.fs)
        
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

    rir_trimmed_norm = normalize_rir(rir_trimmed)

    save_audio(
        cfg.output_dir / cfg.rir_name,
        normalize_for_saving(rir_raw),
        cfg.fs,
    )

    save_audio(
        cfg.output_dir / cfg.trimmed_rir_name,
        normalize_for_saving(rir_trimmed_norm),
        cfg.fs,
    )



    """
    metadata = {
        "use_external_sweep": cfg.use_external_sweep,
        "external_sweep_path": cfg.external_sweep_path if cfg.use_external_sweep else None,
        "recording_path": cfg.recorded_sweep_path2 if cfg.use_external_sweep else cfg.recorded_sweep_path,
        "estimated_lag_samples": lag,
        "estimated_lag_seconds": lag / cfg.fs,
        "recording_clipped": clipped,
        "direct_peak_sample": peak_idx,
        "direct_peak_seconds": peak_idx / cfg.fs,
        "trim_start_sample": trim_start,
        "trim_end_sample": trim_end,
    }

    with open(cfg.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    """

    plot_waveform(sweep_for_plot, cfg.fs, "Reference Sweep")
    plot_waveform(recorded, cfg.fs, "Recorded Signal")
    plot_spectrogram(recorded, cfg.fs, "Recorded Signal Spectrogram")
    plot_rir(rir_raw, cfg.fs, "Raw Extracted RIR")
    plot_rir(rir_trimmed_norm, cfg.fs, "Trimmed + Normalized RIR")
    plot_fft_rir(rir_raw, cfg.fs, 262144, "Frequency Response from RIR")

    edc = energy_curve(rir_trimmed)
    plot_edc(edc, cfg.fs, "Energy Decay Curve")

    print("Done.")
    print(f"Lag: {lag} samples ({lag / cfg.fs:.4f} s)")
    print(f"Clipped: {clipped}")

    show_all()


if __name__ == "__main__":
    main()