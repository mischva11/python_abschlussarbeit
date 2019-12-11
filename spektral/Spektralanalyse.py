# hier passieren ein par dinge, generell ist es die spektren in abhängigkeit der chunks darzustellen

import pandas as pd
from functions.fourier_transform import *
import matplotlib.pyplot as plt
from functions.save_plot import *
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.stats.runs import runstest_1samp


#einlesen der daten
file = "/home/michael/Desktop/Versuch 003/full.csv"
#file = "C:\\Users\\s1656255\\Desktop\\test.csv"
head_names = ["Zeit", "Abstand", "Spannung", "Zahler", "Zustand", "Abstand.2", "Spannung.2",
              "Zahler.2", "Zustand.2", "Abstand.3", "Spannung.3", "Zahler.3", "Zustand.3",
              "Abstand.4", "Spannung.4", "Zahler.4", "Zustand.4", "Abstand.5", "Spannung.5",
              "Zahler.5", "Zustand.5", "Abstand.6", "Spannung.6", "Zahler.6", "Zustand.6", "Temperatur",
              "Strom"]

#zeilen des gesamtfiles zählen
with open(file) as f:
    row_count = sum(1 for line in f)


#erstellung des spektrograms mit verschiedenen chunks (berechnet über modulo)
#teiler ist doppelt so groß wie die verwendeten chunks, da in der csv für jede zeile eine leerzeile existiert
#Anmerkung: unter linux scheint pandas leerzeilen automatisch zu ignorieren.
def plot_spek (modulo = 8, teiler = 50, column = "Temperatur", row_count = row_count, zens=0.5, yzens=0,
               file = file, head_names=head_names):
    plt.plot()
    plt.xlim(0, zens)
    plt.ylim(0, 0.02)
    plt.xlabel(r"\textbf{Frequenz [1/s]}")
    plt.ylabel(r"\textbf{Spektrum [$^{\circ}$C/s]}")
    #plt.title("Fourierspektrum der Aktorendaten")

    #doppelt so hoch wie gewünschte chunks

    chunksize = row_count // teiler
    #chunksize = chunksize + teiler
    counter = 0
    #einzelne chunks einlesen und für jeden chunk die fouriertransformation durchführen
    for chunk in pd.read_csv(file, sep=",", index_col=False, chunksize=chunksize, names=head_names):
        df = chunk
        print(counter)
        #für jeden chunk mit dem  modulo wert die auswertung durchführen und plotten
        if ((counter % modulo) == 0) | (counter == 0) | (counter == (teiler-1)):
            print("Plot bei:", counter)
            abstand_transformiert, freq = transform(df, sample_time=6, column=column)
            freq_zensiert = freq[(freq < zens)]
            #daten werden zensiert, da sonst die pgf's zu groß werden
            abstand_zensiert = abstand_transformiert[:len(freq_zensiert)]
            abstand_doppelt_zensiert = abstand_zensiert[((30000> abstand_zensiert) & (abstand_zensiert > yzens))]
            freq_zensiert = freq_zensiert[((30000 > abstand_zensiert) & (abstand_zensiert > yzens))]
            #plotten
            plt.plot(freq_zensiert[::5], abs(abstand_doppelt_zensiert.real[::5]), alpha=0.4, label=counter+1)

        print(df.tail())
        counter = counter + 1
    plt.legend(loc="best", title="Chunks")
    return plt


#berechnung der standardabweichung der jeweiligen frequenzen
def sd_calc(row_count = row_count, teiler = 50, head_names = head_names, column="Abstand", zens=0.5, yzens=0,
            xlower = 0, xupper = 0, value = 0):
    #setup für den plot
    plt.plot()
    plt.xlabel(r"\textbf{Chunk}")
    plt.ylabel(r"\textbf{Standardabweichung}")

    #chunks ermitteln
    chunksize = row_count // teiler
    #chunksize = chunksize + teiler
    counter = 0
    std_abst=[]
    #für jeden chunk die daten einlesen udn transformieren, durch die frequenzbereiche wird die standardabweichung
    #bestimmt
    for chunk in pd.read_csv(file, sep=",", index_col=False, chunksize=chunksize, names=head_names):
        df = chunk
        #fouriertransformation
        abstand_transformiert, freq = transform(df, sample_time=6, column=column)
        #auch hier wird wieder zensiert um die pgf's kleiner zu halten
        freq_zensiert = freq[(freq < zens)]
        abstand_zensiert = abstand_transformiert[:len(freq_zensiert)]
        abstand_doppelt_zensiert = abstand_zensiert[((30000 > abstand_zensiert) & (abstand_zensiert > yzens))]
        freq_zensiert = freq_zensiert[((30000 > abstand_zensiert) & (abstand_zensiert > yzens))]
        spek_for_sd = abstand_doppelt_zensiert[((xupper > freq_zensiert) & (freq_zensiert > xlower))]
        freq_for_sd = freq_zensiert[((xupper > freq_zensiert) & (freq_zensiert > xlower))]
        spek_for_sd = spek_for_sd[spek_for_sd != 0 & ~np.isnan(spek_for_sd)]

        #standardabweichung für die jeweiligen frequenzbereich berechnen
        if spek_for_sd.size != 0:
            #print("counter", counter)
            std_abst.append(np.std(spek_for_sd.T))
            #print(std_abst)


        counter = counter+1
    #aus irgendeinem grund gibt es einen fehler bei coutner = 45, ab hier wird zensiert
    if counter != 45:
        std_abst = std_abst[:48]
        plt.plot(np.arange(1, (len(std_abst)+1), 1), std_abst, label = "LSL:"+str(xlower)+
                                                                       "  USL:"+str(xupper), color=farbe[i])

    return plt, std_abst



##dieser code wird zum plotten der spektren in chunks verwendet
spek = plot_spek(modulo=20, column="Temperatur", teiler=50)
save_plot("spektrogramm_zeitlicher_ablauf")
spek.show()


## der untere code ist auskommentiert, dieser führt die berechnung der standardabweichungen durch, einfach
## den ganzen block auskommentieren falls gewünscht


# print(row_count)
# #frequenzbereiche festlegen
# xlower=[0.025, 0.065, 0.10, 0.135, 0.175]
# xupper=[0.05, 0.09, 0.125, 0.165, 0.2]
# lr = LinearRegression()
##einheitliche farben für regressionsgerade und daten
# farbe = ["blue", "orange", "green", "red", "black"]
## 5 bereiche also 5 iterationen
# for i in range(5):
#     print(i)
#     sd, std_abstand = sd_calc(xlower=xlower[i], xupper=xupper[i], teiler=50, value=i)
#     Y = np.array(std_abstand).reshape(-1, 1)  # values converts it into a numpy array
#     X = np.arange(1, len(std_abstand) + 1).reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
#     linear_regressor = LinearRegression()  # create object for the class
#     linear_regressor.fit(X, Y)  # perform linear regression
#     Y_pred = linear_regressor.predict(X)  # make predictions
#     sd.plot(X, Y_pred, color = col[i])
#     a = linear_regressor.coef_
#     r_q = linear_regressor.score(X, Y)
#     t = linear_regressor.intercept_
#     print("a:%.7f, t:%.7f, r^2:%.5f" % (a, t, r_q))
#
#     print(runstest_1samp(std_abstand))
#
#     sd.legend(loc="best", title="Frequenzbereich [Hz]:")
#
# save_plot("Frequenzen_im_ueberblick")
# sd.show()
