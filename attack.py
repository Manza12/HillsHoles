import scipy.signal as sig
import scipy.io.wavfile as wav
import scipy.signal.windows as win
import numpy as np
import time
from plot import plot_spectrogram, plot_signal
from synthesis import am_fm_component

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
decay = 2
a = np.exp(-(t-t_0) * decay)
a[t < t_0] = 0
a[t > t_1] = 0

# Smooth amplitude by convolving
SMOOTH_TIME = 0.05  # seconds
SMOOTH_N = int(SMOOTH_TIME * FS)
SMOOTH_SIGMA = 0.0001
SMOOTH_WIN = win.gaussian(SMOOTH_N, SMOOTH_SIGMA*FS) / np.sum(win.gaussian(SMOOTH_N, SMOOTH_SIGMA*FS))
a = sig.convolve(a, SMOOTH_WIN, mode='same')

# Increase attack
a *= np.exp(-(t-t_0)**2) * 4

# Frequency
f = xi * np.ones_like(t)

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
wav.write('attack.wav', FS, x.astype(np.float32))

# Plot
plot_signal(x, fs=FS)
plot_spectrogram(S, tau, omega, vmin=20 * np.log10(EPS), show=True)
