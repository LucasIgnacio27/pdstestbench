# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:39:34 2026

@author: Betterparts
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARÁMETROS DE LA SIMULACIÓN (INCISO A)
# ==========================================
fs = 1000.0         # Frecuencia de muestreo (Hz)
N = 1000            # Cantidad de muestras (1 segundo)
V_R = 2.0           # Rango del ADC: +/- V_R Volts
B = 4               # Bits del cuantizador
kn = 1.0            # Escala para la potencia del ruido aditivo

# Eje de tiempo y frecuencia (unilateral)
t = np.arange(N) / fs
df = fs / N
f_0 = df            # f0 = fs/N = Delta_f
frecuencias = np.fft.rfftfreq(N, d=1/fs)

# ==========================================
# 2. CÁLCULOS DEL ADC Y GENERACIÓN DE SEÑALES
# ==========================================
# Paso de cuantización y potencia teórica del ruido de cuantización
q = (2 * V_R) / (2**B)
P_q = (q**2) / 12.0

# a) Senoidal pura s(t) con potencia unitaria (A = sqrt(2))
s = np.sqrt(2) * np.sin(2 * np.pi * f_0 * t)

# b) Ruido gaussiano incorrelado n(t) con potencia P_n = kn * P_q
sigma_n = np.sqrt(kn * P_q)
n = np.random.normal(0, sigma_n, N)

# c) Señal de entrada al ADC (s_R) y señal cuantizada (s_Q)
s_R = s + n
s_Q = q * np.round(s_R / q)
e_q = s_Q - s_R     # Error de cuantización puro

# ==========================================
# 3. CÁLCULO DE DENSIDADES ESPECTRALES (PSD)
# ==========================================
# Transformadas para espectros
S_Q = np.fft.rfft(s_Q) / N
S_R = np.fft.rfft(s_R) / N
S_analog = np.fft.rfft(s) / N

# Densidades espectrales de potencia en dB (normalizadas)
psd_sQ = 10 * np.log10(np.abs(S_Q)**2 + 1e-12)
psd_sR = 10 * np.log10(np.abs(S_R)**2 + 1e-12)
psd_analog = 10 * np.log10(np.abs(S_analog)**2 + 1e-12)

# Pisos de ruido promediados
piso_analog = 10 * np.log10(np.mean(np.abs(S_R[1:])**2))
piso_digital = 10 * np.log10(np.mean(np.abs(S_Q[1:] - S_analog[1:])**2))

# ==========================================
# 4. GRÁFICOS (INCISO A)
# ==========================================

# Gráfico 1: Señales en el tiempo
plt.figure(figsize=(10, 4.5))
plt.plot(t, s_Q, label=r'$s_Q = Q_{B, V_R}\{s_R\}$ (ADC out)', lw=1.5)
plt.plot(t, s_R, 'g.:', label=r'$s_R = s + n$ (ADC in)', alpha=0.7, markersize=3)
plt.plot(t, s, ':', label=r'$s$ (analog)', color='orange', lw=1.5)
plt.title(f'Señal muestreada por un ADC de {B} bits - $\pm V_R = {V_R}$ V - q = {q:.3f} V', fontweight='bold')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Gráfico 2: Densidad espectral de potencia (PSD)
plt.figure(figsize=(10, 4.5))
plt.plot(frecuencias, psd_sQ, label=r'$s_Q = Q_{B, V_R}\{s_R\}$ (ADC out)')
plt.plot(frecuencias, psd_analog, ':', label=r'$s$ (analog)', color='orange')
plt.plot(frecuencias, psd_sR, 'r:', label=r'$s_R = s + n$ (ADC in)', alpha=0.7)
plt.axhline(piso_analog, color='red', linestyle='--', label=f'$\\overline{{n}} = {piso_analog:.1f}$ dB (piso analog.)')
plt.axhline(piso_digital, color='cyan', linestyle='--', label=f'$\\overline{{n_Q}} = {piso_digital:.1f}$ dB (piso digital)')
plt.title(f'Señal muestreada por un ADC de {B} bits - $\pm V_R = {V_R}$ V - q = {q:.3f} V', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Gráfico 3: Histograma del ruido de cuantización
plt.figure(figsize=(8, 4))
n_bins, bins_edges, _ = plt.hist(e_q, bins=10, density=False, edgecolor='none')
plt.axvline(-q/2, color='red', linestyle='--')
plt.axvline(q/2, color='red', linestyle='--')
plt.hlines(y=len(e_q)/10, xmin=-q/2, xmax=q/2, colors='red', linestyles='--')
plt.title(f'Ruido de cuantización para {B} bits - $\pm V_R = {V_R}$ V - q = {q:.3f} V', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ==========================================
# 5. INCISO B: BARRIDO DE B Y kn
# ==========================================
bits_list = [4, 8, 16]
kn_list = [0.1, 1.0, 10.0]

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)

for i, B_i in enumerate(bits_list):
    for j, kn_j in enumerate(kn_list):
        ax = axes[i, j]
        
        # Parámetros del ADC
        q_i = (2 * V_R) / (2**B_i)
        P_q_i = (q_i**2) / 12.0
        
        # Generación de ruido y señales
        sigma_n_i = np.sqrt(kn_j * P_q_i)
        n_i = np.random.normal(0, sigma_n_i, N)
        s_R_i = s + n_i
        s_Q_i = q_i * np.round(s_R_i / q_i)
        
        # PSD
        S_Q_i = np.fft.rfft(s_Q_i) / N
        S_R_i = np.fft.rfft(s_R_i) / N
        
        psd_sQ_i = 10 * np.log10(np.abs(S_Q_i)**2 + 1e-12)
        psd_sR_i = 10 * np.log10(np.abs(S_R_i)**2 + 1e-12)
        
        piso_analog_i = 10 * np.log10(np.mean(np.abs(S_R_i[1:])**2))
        piso_digital_i = 10 * np.log10(np.mean(np.abs(S_Q_i[1:] - S_analog[1:])**2))
        
        # Plots en sub-ejes
        ax.plot(frecuencias, psd_sQ_i, label=r'$s_Q$', lw=0.8)
        ax.plot(frecuencias, psd_sR_i, 'r:', alpha=0.5, label=r'$s_R$', lw=0.8)
        ax.axhline(piso_analog_i, color='red', linestyle='--', lw=1)
        ax.axhline(piso_digital_i, color='cyan', linestyle='--', lw=1)
        
        ax.set_title(f'B = {B_i} bits | $k_n$ = {kn_j}\nPiso Dig: {piso_digital_i:.1f} dB', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)

for ax in axes[2, :]:
    ax.set_xlabel('Frecuencia [Hz]')
for ax in axes[:, 0]:
    ax.set_ylabel('PSD [dB]')

plt.suptitle('Inciso b): Comparación Espectral para Barrido de $B$ y $k_n$', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()