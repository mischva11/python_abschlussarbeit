# -*- coding: utf-8 -*-
#dieses skript erstellt das beispiel in der einleitung der fouriertransformation

from scipy.fftpack import fft
from functions.save_plot import *
import matplotlib.pyplot as plt
save_space = 'U:\Masterarbeit\Bilder'
# anzahl datenpunkte
N = 600
# abtastrate
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
#erstellen des signals
y =  10*np.cos(20.0 * 2.0*np.pi*x)
#transformation
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)


#plotten des spektrogramms
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlabel(r'\textbf{Frequenz [Hz]}')
plt.ylabel(r'\textbf{Spektrum}')
plt.grid()
save_plot("Fourier_Beispiel")
#plt.show()

#plotten des signals
plt.plot(x, y)
plt.grid()
plt.xlabel(r'\textbf{Zeit [s]}')
plt.ylabel(r'\textbf{Amplitude}')
save_plot("Sinus_Beispiel")
#plt.show()