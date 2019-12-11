#berechnung der fourierkoeffizienten, dieser code ist nicht vollständig von mir. Quelle unbekannt
from __future__ import division
from functions.read_data import *
import numpy as np
import numpy as np
import pylab as py
import scipy
from scipy import integrate

file = "/home/michael/Desktop/Versuch 003/full.csv"
#file = "C:\\Users\\s1656255\\Desktop\\test.csv"
head_names = ["Zeit",	"Abstand", 	"Spannung",	"Zahler",	"Zustand",	"Abstand.2", 	"Spannung.2",
                  "Zahler.2",	"Zustand.2",	"Abstand.3",	"Spannung.3", 	"Zahler.3",	"Zustand.3",
                  "Abstand.4", 	"Spannung.4", 	"Zahler.4",	"Zustand.4",	"Abstand.5", 	"Spannung.5",
                  "Zahler.5",	"Zustand.5",	"Abstand.6", "Spannung.6", 	"Zahler.6",	"Zustand.6",	"Temperatur",
                  "Strom"]

with open(file) as f:
    row_count = sum(1 for line in f)

df = read_data_full(column=["Abstand"], nrows = 2000000)
# Define "x" range.
x = np.arange(0,len(df),1)

# Define "T", i.e functions' period.
T = 320
L = T / 2

# "f(x)" function definition.
def f(x):
    return abs(df["Abstand"] - 4)

# "a" coefficient calculation.
def a(n, L, accuracy = 1000):
    a, b = -L, L
    dx = (b - a) / accuracy
    integration = 0
    for x in np.linspace(a, b, accuracy):
        integration += f(x) * np.cos((n * np.pi * x) / L)
    integration *= dx
    return (1 / L) * integration

# "b" coefficient calculation.
def b(n, L, accuracy = 1000):
    a, b = -L, L
    dx = (b - a) / accuracy
    integration = 0
    for x in np.linspace(a, b, accuracy):
        integration += f(x) * np.sin((n * np.pi * x) / L)
    integration *= dx
    return (1 / L) * integration

# Fourier series.
def Sf(x, L, n = 50):
    a0 = a(0, L)
    sum = np.zeros(np.size(x))
    for i in np.arange(1, n + 1):
        print(i)
        sum += ((a(i, L) * np.cos((i * np.pi * x) / L)) + (b(i, L) * np.sin((i * np.pi * x) / L)))
    return (a0 / 2) + sum

def Sf(x, L, n = 50):
    a0 = a(0, L)
    sum = np.zeros(np.size(x))
    for i in np.arange(1, n + 1):
        print(i)
        sum += ((a(i, L) * np.cos((i * np.pi * x) / L)) + (b(i, L) * np.sin((i * np.pi * x) / L)))
    return (a0 / 2) + sum

# Original signal.
py.plot(x, f(x), linewidth = 1.5, label = 'Signal')

# Approximation signal (Fourier series coefficients).
py.plot(x, Sf(x, L), color = 'red', linewidth = 1.5, label = 'Fourier series', alpha = 0.7)

#py.plot(np.arange(0,len(df)+2000,1), Sf(np.arange(0,len(df)+2000,1), L), '.', color = 'green', linewidth = 1.5, label = 'Fourier series')

# Specify x and y axes limits.
#py.xlim([0, 5])
#py.ylim([-2.2, 2.2])

py.legend(loc = 'upper right', fontsize = '10')

py.show()