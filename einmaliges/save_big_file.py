#zusammenfügen der einzelnen csv_s zu einer großen
import pandas as pd
import os
import glob
import numpy as np
#anmerkung: ich teile die datensätze in 6 teile auf um speicherfehler zu vermeiden, diese sind mit start und end deklariert
#datensatz 004
def read_all(path= "C:\\Users\\s1656255\\Desktop\\Versuch 004", extension = "csv"):
    start = 0
    end = 0
    #alle files einlesen im ausgangsordner
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    result = result[0:51]
    result.sort()
    print(result)
    #alle files einlesen und an das gesamtfile hinzufügen
    for files in result:
        filepath = path + "\\" + files
        df = pd.read_csv(filepath, sep="\s+", skipinitialspace=True, index_col=False, decimal=",", skiprows=None)
        print(end)
        print(len(df))
        print(start)
        print(len(df.iloc[:,0]))
        end = len(df) / 6 + end
        print(len(np.arange(start, end, 1 / 6)))
        df.iloc[:,0] = np.arange(start, end, 1 / 6)[:len(df)]
        start = start + len(df) / 6
        print(df.head)
        #abspeichern des gesamtdatensatzes
        with open(path+"\\full.csv", "a") as f:
            df.to_csv(f, header=False, index=False)
#datensatz 003
def read_all3(path= "C:\\Users\\s1656255\\Desktop\\Versuch 003", extension = "csv"):
    #einlesen der csv namen im ausgangsordner
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    counter = 0
    start = 0
    end = 0
    result = result[0:14]
    result.sort()
    print(result)
    #aus irgendeine grund sind die folgende files komisch formartiert, diese wurden gesondert behandelt
    komisch_formartiert = ["Daten 110.csv", "Daten 113.csv","Daten 117.csv","Daten 122.csv"]
    for files in result:

        if files in komisch_formartiert:
            #zeilen die geskipt werden müssen, werden deklariert
            to_remove = list(range(0, 161))
            #zeile 9 beinhaltet die header, diese bleibt
            to_remove.remove(9)
            filepath = path + "\\" + files
            print(filepath)
            #file einlesen
            df = pd.read_csv(filepath, sep="\s+", skipinitialspace=False, header=0, index_col=False, decimal=",",skiprows=to_remove)
            df = df.drop(labels=["I", "Strom"], axis=1)
            df = df.rename(index=str, columns={"T": "Strom I", "Temperatur": "Temperatur T"})
            df = df.dropna(0)
            print(files)

        else:
            # zeilen die geskipt werden müssen, werden deklariert
            to_remove = list(range(0, 11))
            # zeile 9 beinhaltet die header, diese bleibt
            to_remove.remove(9)
            filepath = path + "\\" + files
            #file einlesen
            df = pd.read_csv(filepath, sep="\s+", skipinitialspace=False, header=0, index_col=False, decimal=",",skiprows=to_remove)
            df = df.drop(labels=["I", "Strom"], axis=1)
            df = df.rename(index=str, columns={"T": "Strom I", "Temperatur": "Temperatur T"})
            df = df.dropna(0)
            print(files)

        end = len(df)/6 + end
        df["Zeit"] = np.arange(start, end, 1/6)
        start = start + len(df)/6

        #abspeichern des files
        with open(path + "\\full.csv", "a") as file:
            df.to_csv(file,header=False, index=False)



#weiß nichtmehr was das ist--------------- vermutlich irgendein experiment


# to_remove = list(range(0, 11))
# to_remove.remove(9)
# df = pd.read_csv("/home/michael/Desktop/Versuch 003/116.csv", sep="\s+", skipinitialspace=False, header=0, index_col=False, decimal=",",
#                  skiprows=to_remove)
# print(df)
# df = df.drop(labels=["I", "Strom"], axis=1)
#
# df = df.rename(index=str, columns={"T": "Strom I", "Temperatur": "Temperatur T"})
# df = df.dropna(0)
# print(df)
#
#
# # with open("/home/michael/Desktop/Versuch 003/full_fail.csv", "a") as file:
# #     df.to_csv(file, header=False, index=False)
# print(df.head())
