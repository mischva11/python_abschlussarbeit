#durchführung einer arima, wird nicht weiter kommentiert, da in arbeit nicht behandelt

import numpy as np
from functions.read_data import *
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
from joblib import Parallel, delayed
import pandas as pd




name="Abstand"
laenge=5000
df = read_data_full(column=[name])

X = df[name].tolist()
X = X[:laenge]
del df
print(X[:5])
size = int(len(X) * 0.90)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
test = test[:50]
# def prima(t):
#     model = ARIMA(history, order=(5, 0, 2))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
#     print("test", test[t])
#     print("predicted=", yhat)
#     return yhat

#
for t in range(len(test)):
    model = ARIMA(history, order=(100, 0, 2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# pool = mp.Pool(4)
# predictions = zip(*pool.map(prima, range(len(test))))
# print(predictions)

# num_cores = mp.cpu_count()
# predictions = Parallel(n_jobs=num_cores)(delayed(prima)(i) for i in range(len(test)))
#
#
# print(test)
# print(predictions)

error = mean_squared_error(test, predictions)
print('Test MSE: %.6f' % error)
# plot
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()