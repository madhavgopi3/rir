from dataclasses import dataclass
from pathlib import Path

@dataclass #dataclasses are used to automatically generate common boilerplate class methods like __init__ for classes that store data.
class MeasurementConfig:
    fs: int = 48000
    sweep_duration: float = 10.0
    f_start: int = 20
    f_end: int = 20000
    f_start2: int = 50
    f_end2: int = 22000
    amplitude: float = 0.8

    pre_silence: float = 0.5
    post_silence: float = 2.0

    rir_pre_trim_ms: float = 5.0
    rir_post_trim_ms: float = 1500.0

    rir_trim_pre_ms: float = 5.0
    rir_min_tail_ms: float = 300.0
    threshold_over_noise_db: float = 15.0
    arrival_smooth_ms: float = 1.0
    tail_smooth_ms: float = 5.0
    safety_offset_ms: float = 30.0

    output_dir: Path = Path("output")
    recorded_sweep_path: Path = Path("recorded/3.wav")
    recorded_sweep_path2: Path = Path("recorded/2.wav")
    external_sweep_path: Path = Path("sweep_48000_50_22000.wav")
    use_external_sweep: bool = False
    
    generated_sweep_name: str = "generated_sweep.wav"
    padded_sweep_name: str = "padded_generated_sweep.wav"
    inverse_sweep_name: str = "inverse_sweep.wav"
    external_inverse_name: str ="external_inverted_sweep.wav"
    rir_name: str = "rir.wav"
    trimmed_rir_name: str = "rir_trimmed.wav"