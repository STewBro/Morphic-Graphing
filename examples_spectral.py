"""
examples_spectral.py
--------------------
Illustrates how morphic graphing can use different analytics
to expose different structure in the same physical law.

Law: damped harmonic oscillator y'' + 2γy' + ω0²y = 0
Analytics: (1) Fourier amplitude  (2) Wavelet amplitude
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2
from numpy.fft import rfft, rfftfreq

# parameters for the simulation
t = np.linspace(0, 10, 4000)
dt = t[1] - t[0]
freqs = np.linspace(1, 5, 80)        # natural frequencies ω0
gammas = np.linspace(0.1, 1.5, 80)   # damping coefficients

# containers for surface data
Z_fourier = np.zeros((len(gammas), len(freqs)))
Z_wavelet = np.zeros_like(Z_fourier)

for i, g in enumerate(gammas):
    for j, w0 in enumerate(freqs):
        y = np.exp(-g * t) * np.cos(2 * np.pi * w0 * t)

        # 1. Fourier analytic: peak amplitude in spectrum
        Y = np.abs(rfft(y))
        f = rfftfreq(len(y), dt)
        Z_fourier[i, j] = Y.max()

        # 2. Wavelet analytic: max amplitude of Morlet transform
        scales = np.arange(1, 64)
        cwtmatr = np.abs(cwt(y, morlet2, scales, w=5))
        Z_wavelet[i, j] = cwtmatr.max()

# plot both morphic surfaces side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
im0 = axes[0].imshow(Z_fourier, extent=[freqs[0], freqs[-1],
                                        gammas[-1], gammas[0]],
                     aspect="auto", cmap="magma")
axes[0].set_title("Spectral analytic (Fourier)")
axes[0].set_xlabel("Natural frequency ω₀")
axes[0].set_ylabel("Damping γ")
fig.colorbar(im0, ax=axes[0], label="Peak amplitude")

im1 = axes[1].imshow(Z_wavelet, extent=[freqs[0], freqs[-1],
                                        gammas[-1], gammas[0]],
                     aspect="auto", cmap="viridis")
axes[1].set_title("Wavelet analytic")
axes[1].set_xlabel("Natural frequency ω₀")
fig.colorbar(im1, ax=axes[1], label="Peak amplitude")

plt.tight_layout()
plt.savefig("assets/damped_oscillator_analytics.png", dpi=160)
plt.close()

print("Saved assets/damped_oscillator_analytics.png")
