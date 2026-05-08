import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_waveform(signal: np.ndarray, fs: int, title: str):
    t = np.arange(len(signal)) / fs
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

"""
def plot_together(signal: np.ndarray, signal2: np.ndarray, fs: int, title: str):
    t = np.arange(len(signal)) / fs
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal, linestyle='--', linewidth = 0.8)
    ax.plot(t, signal2 + 0.1, linestyle='-', linewidth = 0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha = 0.3)
    
    fig.tight_layout()
    return fig


"""

#Now same as plot_waveform. Make changes if needed in the future.
def plot_rir(signal: np.ndarray, fs: int, title: str):
    return plot_waveform(signal, fs, title)

def plot_spectrogram(signal: np.ndarray, fs: int, title: str):
    f, t, mag = spectrogram(signal, fs=fs, nperseg=2048, noverlap=1024, mode="magnitude") # Each FFT window uses 2048 samples. Adjacent windows overlap by 1024 samples.

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t, f, (20*np.log10(mag + 1e-12)), shading="gouraud") #colour will show the magnitude in dB.
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.grid(True, alpha = 0.3)

    fig.colorbar(mesh, ax=ax, label="Magnitude [dB]")
    fig.tight_layout()
    return fig

def plot_edc(edc: np.ndarray, fs: int, title: str = "Energy Decay Curve"):
    t = np.arange(len(edc))/fs
    edc_db = 10 * np.log10(edc + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, edc_db, linewidth = 0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [dB]")
    ax.set_title(title)
    ax.grid(True, alpha = 0.3)

    fig.tight_layout()
    return fig

def plot_fft_rir(h: np.ndarray, fs:int, n_fft: int, title: str): # 65536 because 2^16. freq_resolution = fs/nfft. n_fft is best if it's the next power of 2 greater than len(rir)

    h = np.asarray(h, dtype=np.float64).squeeze()

    H = np.fft.rfft(h, n = n_fft) # H is a complex. Use angle(H) for phase and abs (H) for magnitude.
    freqs = np.fft.rfftfreq(n_fft, d = 1/fs)
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(freqs, magnitude_db)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.set_xlim(20, fs / 2)

    fig.tight_layout()
    return fig

# MATLAB Adaptation Part

"""
    Plot the full centered deconvolved response.
    First half usually contains nonlinear components,
    second half contains the linear impulse response.
"""

def plot_deconvolution_result(
    ir_full: np.ndarray,
    fs: int,
    title: str = "Full Deconvolved Response",
):
    ir_full = np.asarray(ir_full, dtype=np.float64).squeeze()
    t = np.arange(len(ir_full)) / fs
    floor_db = -120.0

    epsilon = 10 ** (floor_db / 20)
    ir_full_db = 20 * np.log10(np.maximum(np.abs(ir_full), epsilon))

    mid_time = (len(ir_full) // 2) / fs

    fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
    ax_wave.plot(t, ir_full)
    ax_wave.axvline(mid_time, linestyle="--")
    ax_wave.set_xlabel("Time [s]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(title)
    ax_wave.grid(True)
    fig_wave.tight_layout()

    fig_db, ax_db = plt.subplots(figsize=(10, 4))
    ax_db.plot(t, ir_full_db)
    ax_db.axvline(mid_time, linestyle="--")
    ax_db.set_xlabel("Time [s]")
    ax_db.set_ylabel("Level [dB]")
    ax_db.set_title(title + " [dB]")
    ax_db.grid(True)
    fig_db.tight_layout()

    return {
        "deconvolution_full": fig_wave,
        "deconvolution_full_db": fig_db,
    }



def plot_linear_and_nonlinear_ir(
    ir_lin: np.ndarray,
    ir_nonlin: np.ndarray,
    fs: int,
):
    """
    Plot linear and nonlinear components separately.
    """
    ir_lin = np.asarray(ir_lin, dtype=np.float64).squeeze()
    ir_nonlin = np.asarray(ir_nonlin, dtype=np.float64).squeeze()

    t_lin = np.arange(len(ir_lin)) / fs
    t_nonlin = np.arange(len(ir_nonlin)) / fs

    fig_nonlin, ax_nonlin = plt.subplots(figsize=(10, 4))
    ax_nonlin.plot(t_nonlin, ir_nonlin)
    ax_nonlin.set_xlabel("Time [s]")
    ax_nonlin.set_ylabel("Amplitude")
    ax_nonlin.set_title("Nonlinear Components")
    ax_nonlin.grid(True)
    fig_nonlin.tight_layout()

    fig_lin, ax_lin = plt.subplots(figsize=(10, 4))
    ax_lin.plot(t_lin, ir_lin)
    ax_lin.set_xlabel("Time [s]")
    ax_lin.set_ylabel("Amplitude")
    ax_lin.set_title("Linear Impulse Response")
    ax_lin.grid(True)
    fig_lin.tight_layout()

    return {
        "nonlinear_ir": fig_nonlin,
        "linear_ir": fig_lin,
    }


def plot_linear_and_nonlinear_db(
    ir_lin: np.ndarray,
    ir_nonlin: np.ndarray,
    fs: int,
    floor_db: float = -120.0,
):
    """
    Plot absolute magnitude in dB for linear and nonlinear components.
    Useful because raw waveform plots can hide low-level structure.
    """
    ir_lin = np.asarray(ir_lin, dtype=np.float64).squeeze()
    ir_nonlin = np.asarray(ir_nonlin, dtype=np.float64).squeeze()

    epsilon = 10 ** (floor_db / 20.0) #Setting a floor to avoid log 0.

    lin_db = 20 * np.log10(np.maximum(np.abs(ir_lin), epsilon))
    nonlin_db = 20 * np.log10(np.maximum(np.abs(ir_nonlin), epsilon))

    t_lin = np.arange(len(ir_lin)) / fs
    t_nonlin = np.arange(len(ir_nonlin)) / fs

    fig_nonlin, ax_nonlin = plt.subplots(figsize=(10, 4))
    ax_nonlin.plot(t_nonlin, nonlin_db)
    ax_nonlin.set_xlabel("Time [s]")
    ax_nonlin.set_ylabel("Level [dB]")
    ax_nonlin.set_title("Nonlinear Components (dB)")
    ax_nonlin.grid(True)
    fig_nonlin.tight_layout()

    fig_lin, ax_lin = plt.subplots(figsize=(10, 4))
    ax_lin.plot(t_lin, lin_db)
    ax_lin.set_xlabel("Time [s]")
    ax_lin.set_ylabel("Level [dB]")
    ax_lin.set_title("Linear Impulse Response (dB)")
    ax_lin.grid(True)
    fig_lin.tight_layout()

    return {
        "non_linear_db": fig_nonlin,
        "linear_ir": fig_lin
    }


def show_all():
    plt.show()
