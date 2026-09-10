import numpy as np                      # Librería para cálculo numérico y manipulación de arreglos
import matplotlib.pyplot as plt          # Librería para la generación de gráficos
import scipy.signal.windows as sp        # Módulo de SciPy para diseñar ventanas digitales
from scipy.signal import welch           # Función de SciPy para calcular la PSD por el método de Welch

# =============================================================================
# 1. PARÁMETROS GENERALES DE LA SIMULACIÓN
# =============================================================================
fs = 1000.0         # Frecuencia de muestreo en Hz (1000 muestras por segundo)
N = 1000            # Largo de la señal temporal y de la FFT (T = N/fs = 1 segundo)
M = 200             # Cantidad de realizaciones Monte Carlo (experimentos independientes)
a = 2               # Rango máximo de desintonía en bins de frecuencia (±2 df)
v_max = np.sqrt(2)  # Amplitud de la sinusoide elegida para que la potencia sea igual a 1 W
snr_db = 15         # Relación Señal a Ruido deseada expresada en decibeles [dB]

df = fs / N                           # Resolución espectral elemental (distancia en Hz entre bins de FFT: 1 Hz)
frec = np.fft.rfftfreq(N, d=1/fs)     # Vector con las frecuencias reales para la FFT unilateral (0 a fs/2)

# =============================================================================
# 2. GENERACIÓN DE REALIZACIONES MONTE CARLO (CON RUIDO)
# =============================================================================
tt = np.arange(N)[:, None]  # Crea un vector de muestras de 0 a N-1 con forma (N, 1) para operar en matrices

# Genera M frecuencias con una pequeña desviación aleatoria uniforme entre -a y +a
fr = np.random.uniform(-a, a, size=(1, M))  # Matriz de dimensión (1, M) con valores aleatorios

# Calcula el valor de frecuencia f0 para cada realización alrededor de fs/4 (250 Hz)
ff = (fs / 4) + fr * df  # Matriz de (1, M) con las frecuencias exactas de cada experimento

# Genera las M sinusoides de forma simultánea aplicando la ecuación temporal en matriz (N, M)
xx_pura = v_max * np.sin(2 * np.pi * ff * tt / fs)  # Dimensión final: N filas por M columnas

# Cálculo de la potencia del ruido para obtener la SNR deseada
p_senial = (v_max**2) / 2                    # Potencia de una sinusoide pura: Vamp^2 / 2 = (sqrt(2))^2 / 2 = 1 W
p_ruido = p_senial / (10**(snr_db / 10))      # Potencia del ruido despejada de la fórmula SNR [dB]

# Genera ruido blanco gaussiano aleatorio con media 0 y la varianza ajustada a la potencia calculada
ruido = np.random.normal(0, np.sqrt(p_ruido), size=(N, M))  # Matriz de ruido con dimensión (N, M)

# Suma el ruido aditivo a las señales sinusoidales puras
xx = xx_pura + ruido  # Matriz final de señales con ruido para trabajar en Monte Carlo

# =============================================================================
# 3. ESTIMACIÓN ESPECTRAL: PERIODOGRAMA VENTANADO (HOLTON CAP. 14)
# =============================================================================

def estimar_periodograma(x, ventana, fs):
    """
    Función que calcula la Densidad Espectral de Potencia (PSD) unilateral
    escalada correctamente por el factor de potencia de la ventana (U).
    """
    N, M = x.shape              # Obtiene las dimensiones de la matriz de entrada (N muestras, M realizaciones)
    w = ventana[:, None]        # Convierte el vector de ventana unidimensional en un vector columna de (N, 1)
    
    # Factor de corrección de energía de la ventana: U = (1/N) * sum(w[n]^2)
    U = np.mean(w**2)           # Calcula el promedio de la ventana al cuadrado
    
    # Multiplica cada señal de la columna por la ventana correspondiente (enventanado temporal)
    x_w = x * w                 # Operación elemento a elemento broadcastizada (N, M)
    
    # Calcula la FFT real unilateral a lo largo del eje 0 (columnas)
    X = np.fft.rfft(x_w, axis=0)  # Devuelve una matriz compleja de (N/2 + 1, M)
    
    # Ecuación del Periodograma Modificado: PSD = |X|^2 / (N * fs * U)
    PSD = (np.abs(X)**2) / (N * fs * U)  # PSD expresada en W/Hz
    
    # Multiplica por 2 todas las frecuencias excepto DC y Nyquist para recuperar la potencia unilateral
    PSD[1:-1, :] *= 2.0         # Cumple con el Teorema de Parseval
    
    return PSD                  # Devuelve la matriz de PSD calculada (frecuencias x realizaciones)

# Genera la ventana Rectangular (vector de N unos)
w_rect = np.ones(N)  

# Llama a la función para calcular el periodograma de las M realizaciones usando la ventana Rectangular
psd_rect = estimar_periodograma(xx, w_rect, fs)

# Genera la ventana de Blackman-Harris de N puntos
w_bmh = sp.blackmanharris(N)  

# Llama a la función para calcular el periodograma de las M realizaciones usando Blackman-Harris
psd_bmh = estimar_periodograma(xx, w_bmh, fs)

# =============================================================================
# 4. MÉTODOS DE REDUCCIÓN DE VARIANZA: ESTIMADOR DE WELCH
# =============================================================================
nperseg = 256  # Define la longitud de cada sub-bloque/segmento en los que se dividirá la señal (256 muestras)

# Aplica la función Welch de SciPy a la primera realización (columna 0 de la matriz xx)
frec_welch, psd_welch = welch(xx[:, 0], fs=fs, window='blackmanharris', 
                              nperseg=nperseg, noverlap=nperseg//2)
# noverlap=nperseg//2 establece una superposición del 50% entre sub-bloques adyacentes

# =============================================================================
# 5. CÁLCULO DE PROMEDIOS Y VARIANZA (ESTADÍSTICA MONTE CARLO)
# =============================================================================
# Estima el valor esperado E{P_hat(f)} calculando el promedio sobre el eje de las M realizaciones (axis=1)
media_psd_rect = np.mean(psd_rect, axis=1)  # Promedio del periodograma rectangular
media_psd_bmh = np.mean(psd_bmh, axis=1)    # Promedio del periodograma Blackman-Harris

# Calcula la varianza estadística del estimador en cada bin de frecuencia
var_psd_rect = np.var(psd_rect, axis=1)      # Varianza del periodograma rectangular
var_psd_bmh = np.var(psd_bmh, axis=1)        # Varianza del periodograma Blackman-Harris

# Conversión de las matrices y promedios a escala logarítmica (dB/Hz) para visualización
psd_rect_db = 10 * np.log10(psd_rect + 1e-12)      # Periodograma rectangular de las M realizaciones en dB
psd_bmh_db = 10 * np.log10(psd_bmh + 1e-12)        # Periodograma Blackman-Harris de las M realizaciones en dB

media_rect_db = 10 * np.log10(media_psd_rect + 1e-12)  # Promedio esperatorio en dB (Rectangular)
media_bmh_db = 10 * np.log10(media_psd_bmh + 1e-12)    # Promedio esperatorio en dB (Blackman-Harris)
psd_welch_db = 10 * np.log10(psd_welch + 1e-12)        # Estimación espectral por Welch en dB

# =============================================================================
# 6. VISUALIZACIÓN COMPLETA
# =============================================================================

# Crea la Figura 1 con dos subgráficos alineados verticalmente que comparten el eje X
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")

# Subgráfico 1: Grafica las M realizaciones del periodograma rectangular en color gris transparente
ax1.plot(frec, psd_rect_db, color='gray', alpha=0.15, lw=0.8)
# Dibuja la curva roja que representa el promedio esperado (Esperanza)
ax1.plot(frec, media_rect_db, color='crimson', lw=2, label=r'Esperanza $E\{\hat{P}(f)\}$ (Monte Carlo)')
ax1.set_title("Periodograma Rectangular: Alto Sesgo Secundario (Fuga) y Alta Varianza", fontweight='bold')
ax1.set_ylabel("PSD [dB/Hz]")            # Titula el eje Y
ax1.set_ylim([-80, 10])                   # Define el límite vertical en dB
ax1.set_xlim([200, 300])                  # Realiza un zoom en la banda de frecuencia de interés [200, 300] Hz
ax1.grid(True, linestyle=':', alpha=0.6) # Enciende la grilla del gráfico
ax1.legend(loc='upper right')             # Ubica la leyenda explicativa arriba a la derecha

# Subgráfico 2: Grafica las M realizaciones del periodograma Blackman-Harris en gris transparente
ax2.plot(frec, psd_bmh_db, color='gray', alpha=0.15, lw=0.8)
# Dibuja la curva azul marino que representa el promedio esperado (Esperanza)
ax2.plot(frec, media_bmh_db, color='navy', lw=2, label=r'Esperanza $E\{\hat{P}(f)\}$ (Monte Carlo)')
ax2.set_title("Periodograma Blackman-Harris: Menor Fuga Espectral", fontweight='bold')
ax2.set_ylabel("PSD [dB/Hz]")            # Titula el eje Y
ax2.set_xlabel("Frecuencia [Hz]")        # Titula el eje X
ax2.set_ylim([-80, 10])                   # Límite vertical
ax2.set_xlim([200, 300])                  # Zoom en frecuencia
ax2.grid(True, linestyle=':', alpha=0.6) # Grilla
ax2.legend(loc='upper right')             # Leyenda

# Crea la Figura 2 para la comparación del efecto del promediado y método de Welch
fig2, ax3 = plt.subplots(1, 1, figsize=(11, 5), layout="constrained")

# Grafica una sola realización del periodograma en gris para evidenciar la varianza
ax3.plot(frec, psd_bmh_db[:, 0], color='lightgray', lw=1, label='Periodograma 1 Realización (Var Alta)')
# Grafica el promedio Monte Carlo de 200 realizaciones
ax3.plot(frec, media_bmh_db, color='navy', lw=1.8, label=r'Promedio Monte Carlo ($M=200$)')
# Grafica la estimación obtenida con el método de Welch en color verde
ax3.plot(frec_welch, psd_welch_db, color='darkgreen', lw=2.2, label=f'Welch (Segmentos={nperseg}, Overlap=50%)')

ax3.set_title("Comparación: Periodograma Simple vs Welch (Reducción de Varianza)", fontweight='bold')
ax3.set_ylabel("PSD [dB/Hz]")            # Etiqueta Y
ax3.set_xlabel("Frecuencia [Hz]")        # Etiqueta X
ax3.set_ylim([-80, 10])                   # Límite vertical
ax3.set_xlim([150, 350])                  # Rango horizontal para observar el ancho de banda del pico
ax3.grid(True, linestyle=':', alpha=0.6) # Grilla
ax3.legend(loc='upper right')             # Leyenda

plt.show()                                # Muestra por pantalla las figuras generadas