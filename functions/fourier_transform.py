import numpy as np

#funktion zur berechnung der fouriertransformation
def transform(data, sample_time, column ):
    #länge
    n = len(data.index)
    #abtastrate
    T = 1/sample_time
    #abstand umrechnen
    data.Abstand = abs(data.Abstand - 4)
    #fft anwendung und normieren
    abstand_transformiert = np.fft.fft(data[column])
    abstand_transformiert = abstand_transformiert[0:int(n / 2)] / n *2
    #frequenz berechnen
    freq = np.linspace(0.0, 1.0 / (2.0 * T), n // 2)
    return abstand_transformiert, freq