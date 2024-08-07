import scipy.signal as sig
import scipy.io.wavfile as wav
import scipy.signal.windows as win
import numpy as np
import time
import matplotlib.pyplot as plt

from plot import plot_spectrogram, plot_signal
from synthesis import am_fm_component


# Design low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    return sig.butter(order, normal_cutoff, btype='low', analog=False)


# Apply low-pass filter
def lowpass_filter(data, cutoff, fs, order=5):
    fil = butter_lowpass(cutoff, fs, order=order)
    y = sig.filtfilt(*fil, data)
    return y


# Global Parameters
EPS = 1e-10
FS = 48000
TIME_RESOLUTION = 0.001  # seconds
FREQUENCY_RESOLUTION = 5  # Hz
WINDOW_DURATION = 0.2  # seconds
WINDOW_SPREAD = 0.005  # seconds

NFFT = int(FS/FREQUENCY_RESOLUTION)
NPERSEG = int(FS*WINDOW_DURATION)
HOP_SIZE = int(FS*TIME_RESOLUTION)
NOVERLAP = NPERSEG - HOP_SIZE
SIGMA = WINDOW_SPREAD * FS

# Signal Parameters
xi = 440  # Hz
t_0 = 0.5  # seconds
t_1 = 1.5  # seconds
duration = 2.  # seconds
volume = 0.2  # no unit

# Time
t = np.linspace(0, duration, int(duration*FS), endpoint=False)

# Amplitude
a = np.ones_like(t)
a[t < t_0] = 0
a[t > t_1] = 0

# Smooth amplitude by convolving
SMOOTH_TIME = 0.2  # seconds
SMOOTH_N = int(SMOOTH_TIME * FS)
SMOOTH_SIGMA = 0.01
SMOOTH_WIN = win.gaussian(SMOOTH_N, SMOOTH_SIGMA*FS) / np.sum(win.gaussian(SMOOTH_N, SMOOTH_SIGMA*FS))
a = sig.convolve(a, SMOOTH_WIN, mode='same')

# Frequency
f_raw = xi * np.ones_like(t) + np.random.normal(0, 10, t.shape)

# Filter frequecy variations

# Parameters for the low-pass filter
cutoff_frequency = 5  # Hz (adjust as needed)
filter_order = 3

# Filter the raw frequency
f = lowpass_filter(f_raw, cutoff_frequency, FS, order=filter_order)

# Amplify frequency variations
f = 5 * (f - xi) + xi

# Plot frequency
plt.figure()
plt.plot(t, f_raw)
plt.plot(t, f)

# Synthesize
s = am_fm_component(a, f, FS)

# Signal
x = volume * s

# STFT
start = time.time()
omega, tau, Z = sig.stft(x, fs=FS, window=('gaussian', SIGMA), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
Z[np.abs(Z) < EPS] = EPS
S = 20 * np.log10(np.abs(Z))
end = time.time()
print('Time to compute spectrogram: %.3f seconds' % (end-start))

# Write to file
wav.write('blur.wav', FS, x.astype(np.float32))

# Plot
plot_signal(x, fs=FS)
plot_spectrogram(S, tau, omega, vmin=20 * np.log10(EPS), show=True)
