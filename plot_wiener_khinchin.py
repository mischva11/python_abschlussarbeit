from functions.read_data import *
from functions.Wiener_Khinchin import *
from functions.save_plot import *
from functions.correlation import *
import matplotlib.pyplot as plt


import math
#variable festlegen
column = "Abstand"
#widerstand berechnen und / oder daten einlesen
if column == "Widerstand":
    df = read_data_full_4(column=["Spannung"])
    df_2 = read_data_full_4(column=["Strom I"])
    df["Widerstand"] = df["Spannung"] / df_2["Strom I"]
    del df_2
else:
    df = read_data_full_4(column=[column])
#theorem durchführen und speicher wieder freigeben
freq, cor_func = wiener_khinchin(df, sample_time=6, column=column)
del freq
del df
#plotten, nach 500 punkten nur noch alle 500 punkte plotten zwecks pgf größe
y = np.arange(0,len(cor_func), 1)
plt.plot(y[0:500], cor_func[0:500],  "b")
plt.plot( y[493:(len(cor_func)-1):15], cor_func[493:(len(cor_func)):15], "b")
#plt.xlim(0,0.2)
plt.xlabel(r'\textbf{Lag}')
plt.ylabel(r'\textbf{Autokorrelationsfunktion}')
plt.xscale("log")
#plt.title("Autokorrelation der Aktorendaten")
#1save_plot("autokorrelation_versuch4")
plt.show()




#cor_func = corr("001", "050", 6, "abstand", 300000)
# plt.plot(cor_func)1
# plt.show()