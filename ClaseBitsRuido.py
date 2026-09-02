# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:54:32 2026

@author: Betterparts
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats

#%% ---------------------- DEFINICION CTES ----------------------

fs = 1000 # Frecuencia muestreo (Hz)
N = 1000 # Cant. muestras
vmax = np.sqrt(2) # Amp max (V)
dc = 0 # Offset (V)
ff = 3 # Frecuencia sinusoidal (Hz)
ph = 0 # Rad
SNR = 40 # dB
Psen = 1 # Potencia media senoide [Watt]
ur = 0 # Media del ruido

k = 4
delta_f = 1

# --- PARÁMETROS DE CUANTIZACIÓN ---
# q representa la resolución del ADC.
# A mayor número de bits (B), menor es el paso q y menor el error de cuantización.
B = 3                            # Número de bits (2^B = 8 niveles)
Vfs = vmax                       # Tensión a escala completa (Full Scale) [V]
qq = (2 * Vfs) / (2**B)          # Paso de cuantización (delta q) [V]

#%% ------------------------- FUNCIONES -------------------------

# Generador de señal senoidal
def gen_sin (vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt = np.arange(0, stop=nn/fs, step=1/fs)
    xx = dc + vmax * np.sin (2 * np.pi * ff * tt + ph)
    return tt, xx


# Generador de ruido
def gen_noise (SNR = SNR, Psen = Psen, ur = ur, nn = N):
    Pr = Psen / (10**(SNR/10))
    desv_est_r = np.sqrt(Pr)
    ruido = np.random.normal(ur, desv_est_r, nn)
    return ruido

#%% ------------------------ MAIN SCRIPT ------------------------

# Invoco la función generadora de senoides
tt, xx = gen_sin( vmax, dc, ff, ph, N, fs)

# Invoco la función generadora de ruido
ruido = gen_noise (SNR, Psen, ur, N)

# Armo manualmente la señal ruidosa
noisy_xx = xx + ruido

# Vector de frecuencias (Resolución espectral df = fs / N)
frec = np.arange(N//2) * fs/N

# --- TEORÍA DE LA FFT ---
# 1. Se multiplica por 1/N para normalizar la amplitud respecto al número de muestras.
# 2. nXX contiene la Transformada Discreta de Fourier de la señal ruidosa.
nXX = 1/N * np.fft.fft(noisy_xx)

plt.close('all')

# =============================================================================
# FIGURA 1: ESPECTRO DE POTENCIA (PSD)
# TEORÍA:
# - Pico en f = 3 Hz: Indica la frecuencia fundamental de la senoide (ff).
# - Piso de ruido (Noise floor): Representa la energía del ruido blanco gaussiano.
# - Factor '2*': Al tomar el espectro unilateral (frecuencias positivas hasta fs/2),
#   se multiplica por 2 para recuperar la potencia de las frecuencias negativas.
# - 10*log10(|X|^2): Convierte la potencia a escala logarítmica en dB.
# =============================================================================
plt.figure(figsize=(9, 5))
plt.plot(frec, 10 * np.log10( 2*(np.abs(nXX[:N//2])) **2 ), label="PSD Unilateral")
plt.title("Espectro de Potencia (PSD) de la Señal Ruidosa")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia / Módulo [dB]")
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# FIGURA 2: ESPECTRO DE FASE
# TEORÍA:
# - En f = 3 Hz: Se observa la fase real de la senoide (ph = 0 rad).
# - En el resto del espectro: Se observa un comportamiento aleatorio entre -pi y pi.
#   Esto ocurre porque el ruido genera amplitudes numéricas muy pequeñas en esas 
#   frecuencias, haciendo que np.angle() evalúe el ángulo de valores complejos ruidosos.
# =============================================================================
plt.figure(figsize=(9, 5))
plt.plot(frec, np.angle(nXX[:N//2]), color='orange', label="Fase espectral")
plt.title("Espectro de Fase de la Señal Ruidosa")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True)
plt.legend()
plt.show()

#%% ----------------------- CUANTIZACIÓN -----------------------

# Proceso de cuantización por redondeo al nivel q más cercano
xx_q = np.round(noisy_xx / qq) * qq

# Error o ruido de cuantización instantáneo
nq = xx_q - noisy_xx

# =============================================================================
# FIGURA 3: SEÑAL RUIDOSA VS SEÑAL CUANTIZADA
# TEORÍA:
# - Muestra la discretización en amplitud realizada por un conversor A/D (ADC).
# - Con B = 3 bits, el rango [-Vfs, Vfs] se divide en 2^3 = 8 niveles continuos.
# - La señal continua se transforma en una forma escalonada.
# =============================================================================
plt.figure(figsize=(9, 5))
plt.plot(noisy_xx, ':x', label="Señal Ruidosa (Entrada ADC)")
plt.plot(xx_q, ':v', label="Señal Cuantizada (Salida ADC)")
plt.title(f"Proceso de Cuantización (B = {B} bits, q = {qq:.3f} V)")
plt.xlabel("Muestras [n]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# FIGURA 4: RUIDO / ERROR DE CUANTIZACIÓN (nq)
# TEORÍA:
# - Definición: nq = xx_q - noisy_xx.
# - Acotación: El error siempre está estrictamente acotado entre [-q/2, q/2].
# - Modelo de ruido: Si la cuantización es fina, nq se modela como ruido blanco 
#   uniformemente distribuido en [-q/2, q/2] con potencia teórica Pq = (q^2) / 12.
# =============================================================================
plt.figure(figsize=(9, 5))
plt.plot(nq, ':x', color='red', label="Error de Cuantización (nq)")
plt.title("Ruido / Error de Cuantización (nq = xx_q - noisy_xx)")
plt.xlabel("Muestras [n]")
plt.ylabel("Error [V]")
plt.grid(True)
plt.legend()
plt.show()

#%% ---------------- TESTS ESTADÍSTICOS Y VALIDACIÓN ----------------

print("\n=============================================================")
print("  VALIDACIÓN ESTADÍSTICA DEL ERROR DE CUANTIZACIÓN (nq)")
print("=============================================================")

# 1. Test de Kolmogorov-Smirnov (Valida si es Distribución Uniforme)
stat_ks, p_val_ks = stats.kstest(nq, 'uniform', args=(-qq/2, qq))
print(f"1. Test K-S (Uniformidad en [-q/2, q/2]): p-valor = {p_val_ks:.4f}")
if p_val_ks > 0.05:
    print("   -> RESULTADO: Se acepta la Distribución Uniforme (p > 0.05).")
else:
    print("   -> RESULTADO: La muestra se desvía de una uniforme pura.")

# 2. Test de Autocorrelación / Incorrelación (Valida si es Ruido Blanco / Delta)
# Evaluamos la correlación entre muestras consecutivas (lag k=1)
r_k1 = np.corrcoef(nq[:-1], nq[1:])[0, 1]
print(f"2. Coeficiente de correlación para lag k=1: {r_k1:.4f}")
if abs(r_k1) < 2 / np.sqrt(N):  # Umbral estadístico al 95% de confianza (2/sqrt(N))
    print("   -> RESULTADO: Muestras incorreladas (Autocorrelación = Delta).")
else:
    print("   -> RESULTADO: Existe dependencia lineal entre muestras.")

# 3. Comparación de Potencia/Varianza
potencia_teorica = (qq**2) / 12
potencia_real = np.var(nq)
print(f"3. Potencia Teórica (q^2 / 12): {potencia_teorica:.6f} W")
print(f"   Potencia Real (Var(nq)):     {potencia_real:.6f} W")
print("=============================================================\n")