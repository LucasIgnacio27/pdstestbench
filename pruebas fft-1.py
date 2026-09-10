# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:22:10 2026

@author: Fede
"""

import sys
from pathlib import Path
directorio_padre = str(Path.cwd().parent)
sys.path.insert(0, directorio_padre)
# import gen_fun as gen
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sp



#%% --------------------------------- Definiciones

fs = 1000
N = 1000
a = 2
v_max = np.sqrt(2)
M = 200


#%% ---------------------------- Ruido cuantizacion

df = fs/N

tt=np.arange(0,N,1)
fr = np.random.uniform(-a,a,M)
f = (fs/4+fr*fs/N)
f = f.reshape(1,M)
tt = tt.reshape(N, 1)
xx = v_max * np.sin(2 * np.pi * f * tt / fs)

ventana = sp.blackmanharris(N)
ventana = ventana.reshape(N,1)
x_v = xx * ventana

X_V = 1/N * np.fft.fft(x_v, axis=0)
X_V = X_V[:(N//2),:]
X_V = ((np.abs(X_V)**2) * 2) / df
X_V_db = 10 * np.log10(X_V)

X = 1/N * np.fft.fft(xx, axis=0)
X = X[:(N//2),:]
X = ((np.abs(X)**2) * 2) / df
X_db = 10 * np.log10(X)

# X_db = 10 * (np.log10((np.abs(X)**2)*2))

frec = np.arange(int(N / 2)) * (fs / N)

#%% ----------------------------------------------ploteos

fig, ax = plt.subplots(1, 1, figsize=(10, 6),layout="constrained")

# --- Subplot 0: Espectro de Magnitud ---
ax.plot(frec,X_db, linewidth=1.2)
ax.set_xlim([240,260])
ax.set_ylim(-60)
ax.set_title("Espectro de potencia normalizado", fontsize=11, fontweight='bold')
ax.set_ylabel("Potencia [dB]", fontsize=10)
ax.set_xlabel("Frecuencia [Hz]")
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.minorticks_on()

fig2, ax = plt.subplots(1, 1, figsize=(10, 6),layout="constrained")

# --- Subplot 0: Espectro de Magnitud ---
ax.plot(frec,X_V_db, linewidth=1.2)
ax.set_xlim([240,260])
ax.set_ylim(-60)
ax.set_title("Espectro de potencia normalizado", fontsize=11, fontweight='bold')
ax.set_ylabel("Potencia [dB]", fontsize=10)
ax.set_xlabel("Frecuencia [Hz]")
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.minorticks_on()

