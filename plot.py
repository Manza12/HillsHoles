import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from typing import List


def format_freq(x, pos, freq_names):
    if pos:
        pass
    n = int(round(x))
    if 0 <= n < len(freq_names):
        return str(round(freq_names[n]))
    else:
        return ""


def format_time(x, pos, time_names):
    if pos:
        pass
    n = int(np.ceil(x))
    if 0 <= n < len(time_names):
        return str(round(time_names[n], 3))
    else:
        return ""


def plot_signal(s, t=None, fs=None, show=False):
    if t is None and fs is None:
        raise ValueError('Either t or fs must be provided.')
    if t is None:
        t = np.arange(len(s)) / fs

    plt.figure()
    plt.plot(t, s)
    plt.title('Signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    if show:
        plt.show()


def plot_signals(signals: List[np.ndarray], t=None, fs=None, show=False):
    if t is None and fs is None:
        raise ValueError('Either t or fs must be provided.')
    if t is None:
        max_duration = max([s.shape[0] for s in signals])
        t = np.arange(max_duration) / fs

    plt.figure()
    for s in signals:
        plt.plot(t, s)

    plt.title('Signals')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    if show:
        plt.show()


def plot_spectrogram(S, tau, omega, ax=None, title=None, vmax=0, vmin=-100, aspect='auto', show=False):
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = 'Spectrogram'

    plt.imshow(S, cmap='afmhot', vmin=vmin, vmax=vmax, aspect=aspect, origin='lower')

    # Format axes
    ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_freq(x, pos, omega)))
    ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_time(x, pos, tau)))

    plt.title(title)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('dB')

    if show:
        plt.show()

    return ax
