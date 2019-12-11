#das script plottet das nur die auswertungen der von der fouriertransformation

import matplotlib.pyplot as plt
from functions.read_data import *
from functions.fourier_transform import *
from functions.save_plot import *
#daten einlesen

df = read_data(dataset="000", nrows=300000)
#fouriertransformation durchführen
abstand_transformiert, freq = transform(df, sample_time=6, column="Abstand")
#daten auf ein wertebereich zum plotten zensieren
zens = 0.5
#plotten
freq_zensiert = freq[(freq < zens)]
print(freq_zensiert[:5])
abstand_zensiert = abstand_transformiert[:len(freq_zensiert)]

plt.plot(freq_zensiert, abs(abstand_zensiert))

plt.xlim(0, zens)
plt.ylim(0.0, 0.25)
plt.xlabel(r"\textbf{Frequenz [1/s]}")
plt.ylabel(r"\textbf{Spektrum [mm/s]}")

# df = read_data(dataset="051", nrows = 300000)
#
# abstand_transformiert, freq = transform(df, sample_time=6, column="Abstand.2")
#
# freq_zensiert = freq[(freq < zens)]
# abstand_zensiert = abstand_transformiert[:len(freq_zensiert)]
#
#plt.plot(freq_zensiert, abs(abstand_zensiert), alpha = 0.7)


#save_plot("Fouriertransformierte_der_Daten")
#plt.show()

save_plot("Fourierspektrum_000_051")
plt.show()