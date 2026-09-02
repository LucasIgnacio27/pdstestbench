# -*- coding: utf-8 -*-
"""
Trabajo Práctico TS1 - Ejercicio 1
Síntesis de Señales y Análisis Espectral Básicos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# PARÁMETROS GENERALES
# ==========================================
N = 1000          # Número total de muestras
f0 = 2000         # Frecuencia fundamental (2 kHz)
Npp = 10          # Muestras por período (al menos 10)
fs = Npp * f0     # Frecuencia de muestreo (20 kHz)
ts = 1 / fs       # Período de muestreo

# Vector de tiempo discreto t = n * ts
tt = np.arange(N) * ts

# Vector de frecuencias asociadas a la FFT
ff = np.fft.fftfreq(N, d=ts)

# ==========================================
# SÍNTESIS DE SEÑALES
# ==========================================

# 1. Senoidal de 2 kHz (al menos 10 puntos/período, A = 1V)
s1 = np.sin(2 * np.pi * f0 * tt)

# 2. Senoidal con P = 2W y desfasada en pi/2 (A = sqrt(2*P) = 2V)
P2 = 2.0
A2 = np.sqrt(2 * P2)
s2 = A2 * np.sin(2 * np.pi * f0 * tt + np.pi/2)

# 3. Ruido Normal (Gaussiano) - Media 0V, Varianza 0.1 W (sigma = sqrt(0.1))
sigma3 = np.sqrt(0.1)
s3 = np.random.normal(loc=0.0, scale=sigma3, size=N)

# 4. Ruido Uniforme - Media 0V, Varianza 0.1 W (A = sqrt(3 * Var) = sqrt(0.3))
A4 = np.sqrt(3 * 0.1)
s4 = np.random.uniform(low=-A4, high=A4, size=N)

# 5. Pulso Rectangular de 2 kHz (P = 1W, Duty Cycle = 50%)
s5 = signal.square(2 * np.pi * f0 * tt, duty=0.5)

# Estrategia de recorrido de señales
senales = [
    ("1. Senoidal 2 kHz (A=1V)", s1),
    ("2. Senoidal desfasada (pi/2) y P=2W", s2),
    ("3. Ruido Normal (Var=0.1W)", s3),
    ("4. Ruido Uniforme (Var=0.1W)", s4),
    ("5. Pulso Rectangular 2 kHz (P=1W)", s5)
]

# ==========================================
# VISUALIZACIÓN EN TIEMPO Y FRECUENCIA
# ==========================================
plt.close('all')

for titulo, x in senales:
    # FFT Normalizada por 1/N para reflejar magnitudes reales
    X_fft = (1 / N) * np.abs(np.fft.fft(x))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(titulo, fontsize=12, fontweight='bold')
    
    # Tiempo: mostramos 50 muestras (5 períodos completos)
    ax1.plot(tt[:50] * 1000, x[:50], 'b-o', markersize=3)
    ax1.set_title("Dominio del Tiempo (primeras 50 muestras)")
    ax1.set_xlabel("Tiempo [ms]")
    ax1.set_ylabel("Amplitud [V]")
    ax1.grid(True)
    
    # Frecuencia centrada en 0 Hz
    ax2.stem(np.fft.fftshift(ff) / 1000, np.fft.fftshift(X_fft), basefmt=" ")
    ax2.set_title("Módulo de la FFT Normalizada |X(f)|")
    ax2.set_xlabel("Frecuencia [kHz]")
    ax2.set_ylabel("Magnitud [V]")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()