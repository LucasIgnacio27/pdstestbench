# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:49:34 2026

@author: Betterparts
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Parámetros
N = 16
k = 2  # frecuencia: 2 ciclos en N muestras

# 1. Señal senoidal
n = np.arange(N)
x = np.sin(2 * np.pi * n * k / N)

# 2. Respuesta al impulso: promedio de 5 muestras (filtro FIR)
h = np.zeros(N)
h[:5] = 1/5  # h[n] = 0.2 para n=0..4, luego ceros

h5 = np.zeros(5)
h5[:5] = 1/5

# -------------------------------------------------------------
# Convolución circular mediante FFT (producto en frecuencia)
# -------------------------------------------------------------
X = np.fft.fft(x)
H = np.fft.fft(h)
y_fft = np.fft.ifft(X * H).real  # parte real (debería ser real)

# -------------------------------------------------------------
# Convolución lineal
# -------------------------------------------------------------
y_h5 = sig.convolve(x, h5, mode='full')

# -------------------------------------------------------------
# Gráficas
# -------------------------------------------------------------
plt.figure(figsize=(12, 8))

plt.plot(x)
plt.plot(y_fft)

plt.figure(figsize=(12, 8))

plt.plot(x)
plt.plot(y_h5)

plt.tight_layout()
plt.show()