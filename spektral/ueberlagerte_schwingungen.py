# -*- coding: utf-8 -*-
#beispiel für die überlagerten schwingungen aus der einleitung, wird nicht weiter kommentiert


from scipy.fftpack import fft
import math
from functions.save_plot import *

save_space = 'U:\Masterarbeit\Bilder'

N = 1000
# sample spacing
T =2 * 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) 
y = y + math.e**(-x)*10*np.cos(12.0 * 2.0*np.pi*x)
y = y + 2*np.sin(7.0 * 2.0*np.pi*x) 
y = y + 1.5*np.cos(50.0 * 2.0*np.pi*x+3)
#white noise with random samples
y = y + np.random.normal(0,2,N)


#y= y + 2.5* scipy.signal.square(4*x * 2.0*np.pi)

yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)


plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlabel(r'\textbf{Frequenz [Hz]}')
plt.ylabel(r'\textbf{Spektrum}')
plt.grid()

save_plot("Fourier_Beispiel_ueberlagert")
plt.show()

plt.plot(x, y)
plt.grid()
plt.xlabel(r'\textbf{Zeit [s]}')
plt.ylabel(r'\textbf{Amplitude}')
save_plot("Schwingung_Beispiel_ueberlagert")
plt.show()
