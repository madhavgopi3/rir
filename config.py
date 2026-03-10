from dataclasses import dataclass
from pathlib import Path

@dataclass #dataclasses are used to automatically generate common boilerplate class methods like __init__ for classes that store data.
class MeasurementConfig:
    fs: int = 48000
    sweep_duration: float = 10.0
    f_start: int = 20
    f_end: int = 20000
    amplitude: float = 0.8

    pre_silence: float = 0.5
    post_silence: float = 2.0

    rir_pre_trim_ms = 5.0
    rir_post_trim_ms = 1500

    output_dir: Path = Path("output")
    recorded_sweep: Path = Path("recorded_sweep.wav")
    
    generated_sweep_name: str = "generated_sweep.wav"
    padded_sweep_name: str = "padded_generated_sweep.wav"
    inverse_sweep_name: str = "inverse_sweep.wav"
    rir_name: str = "rir.wav"
    trimmed_rir_name: str = "rir_trimmed.wav"