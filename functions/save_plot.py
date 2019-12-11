import matplotlib.pyplot as plt
import numpy as np

#eigene funktion um plots abzuspeichern
def save_plot(name ,path = r'/home/michael/Documents/Tex/masterarbeit/bilder/' ):
    #pgf größen und texteigenschaften festlegen
    width = 0.5
    fig_width_pt = 380  
    inches_per_pt = 1.0 / 72.27  
    #plotverhältnis erstellen
    golden_mean = (np.sqrt(5.0) - 1.0) / 3.0  
    fig_width = fig_width_pt * inches_per_pt * width
    figsize = [fig_width, fig_width * golden_mean * 2]
    #setup von matplotlib um in latex die einstellungen zu übernehmen
    latex = {{{{
                        "pgf.texsystem": "pdflatex",        
                        "text.usetex": True,                
                        "font.family": "serif",
                        "font.serif": [],                   
                        "font.monospace": [],
                        "axes.labelsize": 12.7,               
                        "font.size": 12,
                        "legend.fontsize": 8,               
                        "xtick.labelsize": 8,               
                        "ytick.labelsize": 8,
                        "figure.figsize" : figsize
    }

    plt.rcParams.update(latex)
    #abspeichern als pgf
    plt.savefig(path + name + ".pgf")#, dpi=dpi)
