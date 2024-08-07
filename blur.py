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


# Apply low-pass filter with padding
def lowpass_filter(data, cutoff, fs, order=5):
    fil = butter_lowpass(cutoff, fs, order=order)
    pad_len = 10000  # Padding length
    data_padded = np.pad(data, pad_len, mode='reflect')  # Reflect padding
    y = sig.filtfilt(*fil, data_padded)
    return y[pad_len:-pad_len]  # Remove padding


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

# Frequencies
N_FREQ = 5
f_raw = []
for i in range(N_FREQ):
    f_raw.append(xi * np.ones_like(t) + np.random.normal(0, 10, t.shape))

# Filter frequecy variations

# Parameters for the low-pass filter
cutoff_frequency = 5  # Hz (adjust as needed)
filter_order = 3

# Filter the raw frequency
f = []
for i in range(N_FREQ):
    f.append(lowpass_filter(f_raw[i], cutoff_frequency, FS, order=filter_order))

# Amplify frequency variations
for i in range(N_FREQ):
    f[i] = 10 * (f[i] - xi) + xi

# Plot frequencies
plt.figure()
for i in range(N_FREQ):
    plt.plot(t, f[i])

# Synthesize
x = np.zeros_like(t)
s = []
for i in range(N_FREQ):
    s.append(am_fm_component(a, f[i], FS))
    x += s[i] / N_FREQ

# Signal
x *= volume

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
for i in range(N_FREQ):
    plt.plot(t, s[i])
    # wav.write('blur-%d.wav' % i, FS, x.astype(np.float32))
ax = plot_spectrogram(S, tau, omega, vmin=20 * np.log10(EPS))
ax.set_ylim([0, 200])

plt.show()
