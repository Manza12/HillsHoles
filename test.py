import scipy.signal as sig
import scipy.io.wavfile as wav
import numpy as np
import time
from plot import plot_spectrogram
from synthesis import am_fm_component

# Parameters
EPS = 1e-10
FS = 48000
TIME_RESOLUTION = 0.001  # seconds
FREQUENCY_RESOLUTION = 5  # Hz
WINDOW_DURATION = 0.1  # seconds
WINDOW_SPREAD = 0.01  # seconds

NFFT = int(FS/FREQUENCY_RESOLUTION)
NPERSEG = int(FS*WINDOW_DURATION)
HOP_SIZE = int(FS*TIME_RESOLUTION)
NOVERLAP = NPERSEG - HOP_SIZE
SIGMA = WINDOW_SPREAD * FS

# Generate a signal
xi_0 = 1000  # Hz
xi_1 = 1200  # Hz
t_0 = 0.2  # seconds
t_1 = 4.8  # seconds

# smoothing_duration = 0.1  # seconds
# smoothing_sigma = 0.01  # seconds
# smoothing = sig.windows.gaussian(int(smoothing_duration*FS), smoothing_sigma*FS)
# smoothing /= np.sum(smoothing)
# plot_signal(smoothing, fs=FS)

duration = 5.  # seconds
volume = 0.2  # no unit
noise_amplitude = 0.0001  # no unit

t = np.linspace(0, duration, int(duration*FS), endpoint=False)

idx = np.logical_and(t_0 < t, t < t_1)
# a = np.zeros_like(t)
# a[idx] = 1.0
# a_smoothed = sig.convolve(a, smoothing, mode='same')
# plot_signals([a, a_smoothed], fs=FS)
t_center = (t_0 + t_1) / 2
t_spread = (t_1 - t_0) / 12

a = np.exp(-(t-t_center)**2/(2*t_spread**2))

# xi = np.zeros_like(t)
# xi[idx] = xi_0 + (xi_1 - xi_0) * (t[idx] - t_0) / (t_1 - t_0)
f_1 = (xi_0 + xi_1) / 2 + (xi_1 - xi_0) * np.sin(2 * np.pi * 1 * t)
f_2 = (xi_0 + 2*xi_1) / 2 + (2*xi_1 - xi_0) * np.cos(2 * np.pi * 1 * t)

s_1 = am_fm_component(a, f_1, FS)
s_2 = am_fm_component(a, f_2, FS)
# plot_signal(s, fs=FS)

w = noise_amplitude * np.random.randn(len(t))
x = s_1 + s_2 + w
x *= volume / np.max(np.abs(x))

# STFT
start = time.time()
omega, tau, Z = sig.stft(x, fs=FS, window=('gaussian', SIGMA), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
Z[np.abs(Z) < EPS] = EPS
S = 20 * np.log10(np.abs(Z))
end = time.time()
print('Time to compute spectrogram: %.3f seconds' % (end-start))

# Write to file
wav.write('signal.wav', FS, x.astype(np.float32))

# Plot
plot_spectrogram(S, tau, omega, vmin=20 * np.log10(EPS), show=True)
