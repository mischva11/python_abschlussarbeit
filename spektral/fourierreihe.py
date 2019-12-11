#hier ist teilweise code modifiziert übernommen und modifiziert worden
# The MIT License (MIT)
#
# Copyright (c) 2018 by Artem Tartakynov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

#erstellung der prognose mittels der fourierreihe
from __future__ import division
import sklearn.metrics
from functions.save_plot import *
from functions.read_data import *
import numpy as np
import scipy
import pylab as pl
from numpy import fft


#setup für die files
file = "/home/michael/Desktop/Versuch 004/full.csv"
#file = "C:\\Users\\s1656255\\Desktop\\test.csv"
head_names = ["Zeit", "Abstand", "Spannung", "Zaehler", "Abstand.2", "Spannung.2", "Zaehler.2", "Abstand.3",
              "Spannung.3", "Zaehler.3", "Abstand.4", "Spannung.4", "Zaehler.4","Abstand.5", "Spannung.5", "Zaehler.5",
              "Abstand.6", "Spannung.6", "Zaehler.6", "Strom I", "Temperatur"]
#zählen der anzahl der zeilen
with open(file) as f:
    row_count = sum(1 for line in f)

#setup für die anzahl der koeffizienten, sowie die testdaten
nrows = 400000
skiprows = 2000000-nrows
n_predict = 50000
#df = read_data_full(column=["Abstand"], nrows = nrows, skiprows=skiprows)
#df = abs(df-4)



#berechnung der fourierkoeffizienten
def fourierExtrapolation(x, n_predict):
    n = x.size
    # anzahl der fourierkoeffizienten
    n_harm = 3000000
    t = np.arange(0, n)
    # trenddetektion
    p = np.polyfit(t, x, 1)
    # trendentfernung
    x_notrend = x - p[0] * t
    # trendentfernung in x domäne
    x_freqdom = fft.fft(x_notrend)
    #frequenzen berechnen
    f = fft.fftfreq(n)
    indexes = range(n)
    # sort indexes by frequency, lower -> higher
    indexes = list(range(n))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        #berechnung der amplitude
        ampli = np.absolute(x_freqdom[i]) / n
        #berechnung der phase
        phase = np.angle(x_freqdom[i])
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

#plot range aufsetzen
start_plot = nrows - 10000
end_plot = nrows + 10000
slice = 50
#val_data = read_data_full_4(column=["Abstand"], skiprows = nrows, nrows = n_predict)
#val_data = abs(val_data-4)
def main():
    vals = ["Abstand", "Abstand.2", "Abstand.3", "Abstand.4", "Abstand.5", "Abstand.6"]
    vals = ["Abstand.6"]
    for val in vals:
        #daten laden
        df = read_data_full_4(column=[val], nrows=nrows, skiprows=skiprows)
        df = abs(df - 4)
        #validierungsdaten laden
        val_data = read_data_full_4(column=[val], skiprows=nrows, nrows=n_predict)
        val_data = abs(val_data - 4)
        x = df[val]
        x = np.array(x)
        #fourierreihe anwenden
        extrapolation = fourierExtrapolation(x, n_predict)
        #nicht mehr ganz sicher was das macht, vermutlich den plot auf ein fenster anpassen

        #pl.plot(np.arange(0, extrapolation.size)[nrows::slice], extrapolation[nrows::slice], 'r', label='extrapolation')
        #pl.plot(np.arange(0, x.size)[start_plot::slice], x[start_plot::slice], 'b', label='x', linewidth=3)
        #pl.plot(np.arange(nrows, nrows+n_predict)[start_plot::slice], val_data[start_plot::slice], color="green", alpha = 0.7)
        #pl.legend()
        #pl.show()
        #save_plot("Prediction_Fourier_series")
        print(val)

        #fehlerabweichung ausgeben
        print(sklearn.metrics.mean_squared_error(val_data, extrapolation[nrows:]))


if __name__ == "__main__":
    main()
