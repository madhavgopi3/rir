import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_waveform(signal: np.ndarray, fs: int, title: str):
    t = np.arange(len(signal)) / fs
    
    plt.figure(figsize=[10,4])
    plt.plot(t, signal, linewidth = 0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()

"""
def plot_together(signal: np.ndarray, signal2: np.ndarray, fs: int, title: str):
    t = np.arange(len(signal)) / fs
    
    plt.figure(figsize=[10,4])
    plt.plot(t, signal, linestyle='--', linewidth = 0.8)
    plt.plot(t, signal2 + 0.1, linestyle='-', linewidth = 0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.show()

"""

#Now same as plot_waveform. Make changes if needed in the future.
def plot_rir(signal: np.ndarray, fs: int, title: str):
    plot_waveform(signal, fs, title)

def plot_spectrogram(signal: np.ndarray, fs: int, title: str):
    f, t, mag = spectrogram(signal, fs=fs, nperseg=2048, noverlap=1024, mode="magnitude") # Each FFT window uses 2048 samples. Adjacent windows overlap by 1024 samples.

    plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f, (20*np.log10(mag + 1e-12)), shading="gouraud") #colour will show the magnitude in dB.
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()

def plot_edc(edc: np.ndarray, fs: int, title: str = "Energy Decay Curve"):
    t = np.arange(len(edc))/fs
    edc_db = 10 * np.log10(edc + 1e-12)

    plt.figure(figsize=[10,4])
    plt.plot(t, edc_db, linewidth = 0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Level [dB]")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()

def plot_fft_rir(h: np.ndarray, fs:int, n_fft: int, title: str): # 65536 because 2^16. freq_resolution = fs/nfft. n_fft is best if it's the next power of 2 greater than len(rir)

    h = np.asarray(h, dtype=np.float64).squeeze()

    H = np.fft.rfft(h, n = n_fft) # H is a complex. Use angle(H) for phase and abs (H) for magnitude.
    freqs = np.fft.rfftfreq(n_fft, d = 1/fs)
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, magnitude_db)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Frequency Response from RIR")
    plt.grid(True, which="both")
    plt.xlim(20, fs / 2)
    plt.show()

# MATLAB Adaptation Part

def plot_deconvolution_result(
    ir_full: np.ndarray,
    fs: int,
    title: str = "Full Deconvolved Response",
):
    """
    Plot the full centered deconvolved response.
    First half usually contains nonlinear components,
    second half contains the linear impulse response.
    """
    ir_full = np.asarray(ir_full, dtype=np.float64).squeeze()
    t = np.arange(len(ir_full)) / fs
    floor_db = -120.0

    epsilon = 10 ** (floor_db/20)

    ir_full_db = 20 * np.log10(np.maximum(np.abs(ir_full),epsilon))

    plt.figure(figsize=(10, 4))
    plt.plot(t, ir_full)
    plt.axvline((len(ir_full) // 2) / fs, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, ir_full_db)
    plt.axvline((len(ir_full_db) // 2) / fs, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Level [dB]")
    plt.title(title + "[dB]")
    plt.grid(True)
    plt.tight_layout()



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

    plt.figure(figsize=(10, 4))
    plt.plot(t_nonlin, ir_nonlin)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Nonlinear Components")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t_lin, ir_lin)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Linear Impulse Response")
    plt.grid(True)
    plt.tight_layout()


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

    plt.figure(figsize=(10, 4))
    plt.plot(t_nonlin, nonlin_db)
    plt.xlabel("Time [s]")
    plt.ylabel("Level [dB]")
    plt.title("Nonlinear Components (dB)")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t_lin, lin_db)
    plt.xlabel("Time [s]")
    plt.ylabel("Level [dB]")
    plt.title("Linear Impulse Response (dB)")
    plt.grid(True)
    plt.tight_layout()


def show_all():
    plt.show()
