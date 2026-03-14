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
    plt.tight_layout

#Now same as plot_waveform. Make changes if needed in the future.
def plot_rir(signal: np.ndarray, fs: int, title: str):
    plot_waveform(signal, fs, title)

def plot_spectrogram(signal: np.ndarray, fs: int, title: str):
    f, t, mag = spectrogram(signal, fs=fs, nperseg=2048, noverlap=1024, mode="magnitude") # Each FFT window uses 2048 samples. Adjacent windows overlap by 1024 samples.

    plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f, (20*np.log10(signal + 1e-12)), shading="gouraud") #colour will show the magnitude in dB.
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()

def plot_edc(edc: np.ndarray, fs: int, title: str = "Energy Decay Curve"):
    t = np.arange(len(edc))/fs
    edc_db = 10 * np.log10(edc + 1e-12)

    plt.figure(figsize=[10,4])
    plt.plot(t, edc, linewidth = 0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Level [dB]")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout

def show_all():
    plt.show()
