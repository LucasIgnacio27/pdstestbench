import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
bins = np.arange(N)

# 1. Filtro en frecuencia H[k] = 1 - (-1)^k
h = np.zeros(N)
h[0] = 1.0
h[4] = -1.0
H_fft = np.fft.fft(h, n=N)

# 2. Entradas x[n] respetando la ventana de 8 puntos
# Inciso A: Coseno k0 = 1
x_a = np.cos(2 * np.pi * 1 * n / N)

# Inciso B: Exponencial (1/2)^n * u[n]
x_b = (0.5) ** n

# Inciso C: Pulso con envolvente circular x[n] = [1, 1, 0, 0, 0, 0, 0, 1]
x_c = np.zeros(N)
x_c[0] = 1.0  # n = 0
x_c[1] = 1.0  # n = 1
x_c[7] = 1.0  # n = -1 en la DFT de 8 puntos (n = -1 + 8 = 7)

# 3. Filtrado Y[k] = X[k] * H[k]
Y_a_fft = np.fft.fft(x_a, n=N) * H_fft
Y_b_fft = np.fft.fft(x_b, n=N) * H_fft
Y_c_fft = np.fft.fft(x_c, n=N) * H_fft

# 4. Función de reporte numérico y visual
def evaluar_inciso(titulo, Y_fft, fig_num):
    mag = np.abs(Y_fft)
    fase = np.angle(Y_fft)
    
    # Limpieza de residuo por precisión flotante
    mag[np.isclose(mag, 0, atol=1e-12)] = 0.0
    fase[mag == 0] = 0.0

    print(f"\n==========================================")
    print(f"   DFT Y[k] - {titulo.upper()}")
    print(f"==========================================")
    print(f"{'Bin (k)':<8} | {'|Y[k]|':<12} | {'Fase (rad)':<12}")
    print("-" * 40)
    for k in range(N):
        print(f"{k:<8} | {mag[k]:<12.4f} | {fase[k]:<12.4f}")

    plt.figure(fig_num, figsize=(7.5, 3.8))
    plt.stem(bins, mag, basefmt=" ")
    for k, val in zip(bins, mag):
        plt.annotate(f'{val:.2f}', xy=(k, val), xytext=(0, 4),
                     textcoords="offset points", ha='center', va='bottom', 
                     fontsize=9, fontweight='bold')

    plt.title(f"Módulo |Y[k]| - {titulo}", fontsize=11, fontweight='bold')
    plt.xlabel("Bin de Frecuencia (k)")
    plt.ylabel("Magnitud")
    plt.xticks(bins)
    plt.ylim(0, max(max(mag) * 1.2, 1.0))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

evaluar_inciso("Inciso A (Coseno k0=1)", Y_a_fft, 1)
evaluar_inciso("Inciso B (Exponencial)", Y_b_fft, 2)
evaluar_inciso("Inciso C (Pulso x[7]=1)", Y_c_fft, 3)

plt.show()