import numpy as np


def am_fm_component(a: np.ndarray, f: np.ndarray, fs, phi_0: float = 0.0) -> np.ndarray:
    """
    Generate an AM-FM component.
    :param a: amplitude vector
    :param f: frequency vector
    :param fs: sampling frequency
    :param phi_0: initial phase
    :return: AM-FM component
    """
    return a * np.sin(2 * np.pi * np.cumsum(f) / fs + phi_0)


def am_tm_component(a: np.ndarray, t: np.ndarray, fs, t_0: float = 0.0) -> np.ndarray:
    """
    Generate an AM-FM component.
    :param a: amplitude vector
    :param t: time vector
    :param fs: sampling frequency
    :param t_0: initial time
    :return: AM-FM component
    """
    return np.real(np.fft.rfft(a * np.exp(1j * 2 * np.pi * np.cumsum(t) / fs + t_0)))
