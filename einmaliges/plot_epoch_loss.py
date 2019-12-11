import pandas as pd
import matplotlib.pyplot as plt
from functions.save_plot import save_plot
#daten einlesen - ausgabe aus tensorboard
df = pd.read_csv("/home/michael/Documents/masterarbeit_pycharm/einmaliges/run-log20-tag-epoch_val_loss.csv")

print(df)


#plotten des epochenverlusts
plt.plot(df.Step, df.Value)
plt.ylabel(r"\textbf{Epochen Verlust}")
plt.xlabel(r"\textbf{Epoche}")
save_plot("epoch_loss_val")
