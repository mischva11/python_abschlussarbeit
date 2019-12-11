#dieses script plottet die verschiedenen aktivierungsfunktionen im vergleich - keine weitere kommentierung

from functions.save_plot import *

x = np.arange(-3, 3, 00.1)

#tanh
y = np.tanh(x)

plt.plot(x,y, label = "Tangens hyperbolicus")


#relu
y = (abs(x) + x) / 2

plt.plot(x,y, label="ReLU")


#sigmoid


y= 1 / (1 + np.exp(-2*x))

plt.plot(x,y, label="Sigmoid")

plt.ylim(-1,1.5)
plt.xlim(-1.5, 1.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
save_plot("aktivierungsfunktionen")
#plt.show()

