#MATLAB Code Adaptation

from __future__ import annotations
import numpy as np


def _as_1d_float(x: np.ndarray, name: str) -> np.ndarray:
    """
    Convert input to a 1D float64 NumPy array.
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return x


def deconvolve_sweep_response(
    sweep_response: np.ndarray,
    inverse_sweep: np.ndarray,
    n_fft: int | None = None,
    center_result: bool = True,
) -> np.ndarray:
    """
    Deconvolve a recorded swept-sine response with an inverse sweep.

    Parameters
    ----------
    sweep_response : np.ndarray
        Recorded sweep response from the system / room.
    inverse_sweep : np.ndarray
        Time-domain inverse sweep filter.
    n_fft : int | None
        FFT length. If None, uses a power-of-two length >= len(response)+len(inverse)-1.
    center_result : bool
        If True, circularly shift by half the FFT length, similar to the MATLAB code.

    Returns
    -------
    ir : np.ndarray
        Full deconvolved response. If center_result=True, nonlinear components
        tend to appear in the first half and linear response in the second half.
    """
    sweep_response = _as_1d_float(sweep_response, "sweep_response")
    inverse_sweep = _as_1d_float(inverse_sweep, "inverse_sweep")

    full_len = len(sweep_response) + len(inverse_sweep) - 1

    if n_fft is None:
        # Next power of two for efficient FFT
        n_fft = 1 << int(np.ceil(np.log2(full_len))) # 1 << means 2 **. This line finds the ceiling of the power of 2 that full_len can be expressed as.
        # We do it because fft if fastest for powers of 2.

    response_fft = np.fft.fft(sweep_response, n=n_fft)
    inverse_fft = np.fft.fft(inverse_sweep, n=n_fft)

    ir = np.real(np.fft.ifft(response_fft * inverse_fft)) # Multiplication in freq domain = conv in time domain.

    if center_result:
        ir = np.roll(ir, n_fft // 2) #Circular shift from half. Eg: If ir = [1, 2, 3, 4, 5, 6, 7, 8], it becomes: [5, 6, 7, 8, 1, 2, 3, 4]
        """
        Because:
        In Farina’s method:
        Nonlinear harmonics appear earlier in time
        Linear IR appears later
        """
    return ir


def split_linear_nonlinear(ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a centered deconvolved response into nonlinear and linear parts.

    This follows the same idea as the MATLAB code:
    - first half  -> nonlinear distortion components
    - second half -> linear impulse response

    Parameters
    ----------
    ir : np.ndarray
        Centered full deconvolved response.

    Returns
    -------
    ir_lin : np.ndarray
        Linear impulse response portion.
    ir_nonlin : np.ndarray
        Nonlinear distortion portion.
    """
    ir = _as_1d_float(ir, "ir")

    mid = len(ir) // 2
    ir_nonlin = ir[:mid]
    ir_lin = ir[mid:]

    return ir_lin, ir_nonlin


def extract_ir_sweep(
    sweep_response: np.ndarray,
    inverse_sweep: np.ndarray,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full swept-sine deconvolution and separation.

    Parameters
    ----------
    sweep_response : np.ndarray
        Recorded sweep response.
    inverse_sweep : np.ndarray
        Time-domain inverse sweep.
    n_fft : int | None
        FFT length. If None, auto-selects one.

    Returns
    -------
    ir_lin : np.ndarray
        Linear impulse response.
    ir_nonlin : np.ndarray
        Nonlinear distortion portion.
    ir_full : np.ndarray
        Full centered deconvolved response.
    """
    ir_full = deconvolve_sweep_response(
        sweep_response=sweep_response,
        inverse_sweep=inverse_sweep,
        n_fft=n_fft,
        center_result=True,
    )

    ir_lin, ir_nonlin = split_linear_nonlinear(ir_full)
    return ir_lin, ir_nonlin, ir_full