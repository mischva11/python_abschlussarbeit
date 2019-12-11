import numpy as np

#wiener khinchin theorem, die delete kommandos sind um den speicher leerzubekommen.
#es entstehen sonst memory errors

def wiener_khinchin(data, sample_time, column):

    #setup
    data = data[column]
    N = len(data.index)
    T = 1 / sample_time
    cor_func = []
    data = data - data.mean()
    n=N
    #fft fürs theorem
    transformiert = np.fft.fft(data, n * 2)
    del data
    #eigentliches theorem
    power_trans = transformiert * np.conjugate(transformiert)
    del transformiert
    #zuschneiden
    power_func = np.fft.ifft(power_trans)
    power_func = power_func[:n]
    #korrenationsfunktion erstellen
    cor_func = np.append(cor_func, np.real(power_func))
    del power_func

    #normieren
    for i in range(N):
        cor_func[i] = cor_func[i] / (n-i)
    cor_func = cor_func / cor_func[0]

    #frequenz wird berechnet, aber generell nicht benötigt. eigentlich unnötig
    freq = np.linspace(0.0, 1.0 / (2.0 * T), N * 2)
    #rückgabe
    return freq, cor_func




