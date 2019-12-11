#dieses script erstellt einen ausschnitt der daten
from functions.read_data import *
from functions.save_plot import *

#gewünscht anzahl der daten
number_of_data = 600

#einlesen
df=read_data_full(nrows=number_of_data, skiprows=20000000)
print(df.head())

#zuschneiden auf die gewünschte anzahl der daten
spannung = pd.to_numeric(df.Spannung[:number_of_data])
zeit = pd.to_numeric(df.Zeit[:number_of_data])
abstand = pd.to_numeric(df.Abstand[:number_of_data])

#abstand absolut darstellen mit positiven werten
abstand = abs(abstand - 4)
temperatur = pd.to_numeric(df.Temperatur[:number_of_data])

#plotten
plt.subplot(311)
plt.plot(zeit, spannung)
plt.ylabel(r"\textbf{Spannung [V]}")
plt.xlabel(r"\textbf{Zeit [s]}")
plt.subplot(312)
plt.subplots_adjust(hspace=0.5)
plt.plot(zeit, abstand)
plt.ylabel(r"\textbf{Auslenkung [mm]}")
plt.xlabel(r"\textbf{Zeit [s]}")
plt.subplot(313)
plt.plot(zeit, temperatur)
plt.ylabel (r"\textbf{Temperatur [$^{\circ}$C]}")
plt.xlabel (r"\textbf{Zeit [s]}")
save_plot("sample_data_ende")
plt.show()
