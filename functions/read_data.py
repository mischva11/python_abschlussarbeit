import pandas as pd
import itertools

#funktionen zum einlesen der einzelnen datensätze, je nach datensatz werden andere header verwendet.
def read_data(path = "~/Desktop/Versuch 004/Daten 000.csv", dataset = False, nrows = None, skiprows = None):
    if dataset == False:
        head_names = ["Zeit", "Abstand", "Spannung", "Zaehler", "Abstand.2", "Spannung.2", "Zaehler.2", "Abstand.3",
                      "Spannung.3", "Zaehler.3", "Abstand.4", "Spannung.4", "Zaehler.4",
                      "Zaehler.5", "Abstand.5", "Spannung.5", "Zaehler.6", "Abstand.6", "Spannung.6","Strom I", "Temperatur"]
        df = pd.read_csv(path, sep="\s+", skipinitialspace=True, names=head_names, index_col=False, decimal=",",
                         nrows=nrows, skiprows = None)
        return df
    else:
        path = "~/Desktop/Versuch 004/Daten " + dataset + ".csv"
        head_names = ["Zeit", "Abstand", "Spannung", "Zaehler", "Abstand.2", "Spannung.2", "Zaehler.2", "Abstand.3",
                      "Spannung.3", "Zaehler.3", "Abstand.4", "Spannung.4", "Zaehler.4",
                      "Zaehler.5", "Abstand.5", "Spannung.5", "Zaehler.6", "Abstand.6", "Spannung.6","Strom I", "Temperatur"]
        df = pd.read_csv(path, sep="\s+", skipinitialspace=True, names=head_names, index_col=False, decimal=",",
                         nrows=nrows, skiprows = None)
        return df


def read_data3(path="~/Desktop/Versuch 003/110.csv", dataset=False, nrows=None, column=None):
    if (dataset == False) | (dataset == "110"):
        to_remove = list(range(0, 161))
        to_remove.remove(9)
        df = pd.read_csv(path, sep="\s+", skipinitialspace=False, header=0, index_col=False, decimal=",", nrows=nrows,
                         skiprows=to_remove, usecols=column)
        if column == None:
            df = df.drop(labels=["I", "Strom"], axis=1)
            df = df.rename(index=str, columns={"T":"Strom I", "Temperatur":"Temperatur T"})
    else:
        to_remove = list(range(0, 11))
        to_remove.remove(9)
        path = "~/Desktop/Versuch 003/" + dataset + ".csv"
        df = pd.read_csv(path, sep="\s+", skipinitialspace=False, header=0, index_col=False, decimal=",", nrows=nrows,
                         skiprows=to_remove, usecols=column)
        if column == None:
            df = df.drop(labels=["I", "Strom"], axis=1)
            df = df.rename(index=str, columns={"T":"Strom I", "Temperatur":"Temperatur T"})

    print(df[:5])
    return df

#diese beiden funktionen lesen die erstellten gesamtdatensätze ein
def read_data_full(column = None, nrows=None, skiprows=None):
    path = "/home/michael/Desktop/Versuch 003/full.csv"
    # file = "C:\\Users\\s1656255\\Desktop\\test.csv"
    head_names = ["Zeit",	"Abstand", 	"Spannung",	"Zahler",	"Zustand",	"Abstand.2", 	"Spannung.2",
                  "Zahler.2",	"Zustand.2",	"Abstand.3",	"Spannung.3", 	"Zahler.3",	"Zustand.3",
                  "Abstand.4", 	"Spannung.4", 	"Zahler.4",	"Zustand.4",	"Abstand.5", 	"Spannung.5",
                  "Zahler.5",	"Zustand.5",	"Abstand.6", "Spannung.6", 	"Zahler.6",	"Zustand.6",	"Temperatur",
                  "Strom"]

    df = pd.read_csv(path, sep=",", index_col=False, names=head_names, usecols=column, nrows=nrows, skiprows=skiprows)
    print(df[:5])
    return df


def read_data_full_4(path = "~/Desktop/Versuch 004/full.csv", dataset = False, nrows = None, skiprows = None, column = None):
    if dataset == False:
        head_names = ["Zeit", "Abstand", "Spannung", "Zaehler", "Abstand.2", "Spannung.2", "Zaehler.2", "Abstand.3",
              "Spannung.3", "Zaehler.3", "Abstand.4", "Spannung.4", "Zaehler.4","Abstand.5", "Spannung.5", "Zaehler.5",
              "Abstand.6", "Spannung.6", "Zaehler.6", "Strom I", "Temperatur"]
        df = pd.read_csv(path, skipinitialspace=True, names=head_names, index_col=False, sep=",",
                         nrows=nrows, skiprows = None, usecols=column)
        return df