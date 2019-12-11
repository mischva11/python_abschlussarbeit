# einige Teile dieses Codes wurden auf Basis eines öffentlichen Codes erstellt. Um der MIT Lizenz genüge zu tun:
# The MIT License (MIT)
#
# Copyright (c) 2016 by Magnus Erik Hvass Pedersen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from functions.read_data import *
from functions.batch_gen import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, TimeDistributed
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.layers import Dropout

    # Maybe use lower init-ranges.
init = RandomUniform(minval=-0.05, maxval=0.05)
gpu_options = tf.GPUOptions(allow_growth=True)
#session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


#def loss function
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

#soll gefittet werden oder nur validiert
fit = True
#def batch size
#batchsizes festlegen
batch_size = 64
sequence_length = 768

#data setup
nrow=2200000
result_2 =[]

cols = ["Abstand", "Abstand.2", "Abstand.3", "Abstand.4", "Abstand.5", "Abstand.6"]
sps = ["Spannung", "Spannung.2", "Spannung.3", "Spannung.4", "Spannung.5", "Spannung.6"]
cols = ["Abstand"]
counter = 0
#für wie viele aktoren soll der datensatz durchlaufen
for col in cols:
    #daten einlesen
    #df = read_data(nrows=10000)
    col = cols[counter]
    sp = sps[counter]
    df = read_data_full(column=[sp, "Strom", col, "Zeit"], nrows=nrow)
    counter = counter + 1
    df.rename(columns={col: "Abstand"}, inplace=True)
    print(df.head())
    df["Widerstand"] = df[sp] / df["Strom"]
    print(df.head())
    df = df.drop([sp, "Strom"], axis=1)
    df["Abstand"] = abs(df["Abstand"] - 4)
    print(df.head(9))
    print(df.shape)

    # path = "C:/Users/s1656255/Desktop/full.csv"
    # head_names = ["Zeit", "Abstand", "Spannung", "Zahler", "Zustand", "Abstand.2", "Spannung.2",
    #               "Zahler.2", "Zustand.2", "Abstand.3", "Spannung.3", "Zahler.3", "Zustand.3",
    #               "Abstand.4", "Spannung.4", "Zahler.4", "Zustand.4", "Abstand.5", "Spannung.5",
    #               "Zahler.5", "Zustand.5", "Abstand.6", "Spannung.6", "Zahler.6", "Zustand.6", "Temperatur",
    #               "Strom"]
    # df = tf.data.experimental.make_csv_dataset(
    #     path,
    #     batch_size,
    #     column_names=head_names,
    # )

    #
    #
    # df = read_data(nrows=53760)
    # print(df.head(9))
    # df["Widerstand"] = df["Spannung"] / df["Strom I"]
    #df["Abstand"] = abs(df["Abstand"] - 4)

    #zielvariablen festlegen
    target_var = ["Abstand"]
    used_var = ["Zeit", "Abstand"] #"Zeit",
    target_names = target_var
    df = df[used_var]
    #wie viel schritte sollen prediktet werden
    shift_steps = 100000
    df_target = df[target_var].shift(-shift_steps)

    ##neural network daten setup
    #data
    x_data = df.values[0:-shift_steps]
    print("Shape:", x_data.shape)
    #target
    y_data = df_target.values[:-shift_steps]
    print("Shape:", y_data.shape)
    #length of data
    num_data = len(x_data)
    #fraction of the data-set that will be used for the training-set
    train_split = 0.9
    #number of observations in the training-set
    num_train = int(train_split * num_data)
    #number of observations
    num_test = num_data - num_train
    #input signals
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    #output signals
    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]
    #number of input signals
    num_x_signals = x_data.shape[1]
    #number of output signals
    num_y_signals = y_data.shape[1]

    #scaling
    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.fit_transform(x_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    #batches
    # del x_data
    # del y_data
    del df
    generator = batch_generator(batch_size, sequence_length, x_train_scaled, y_train_scaled, num_train, num_y_signals, num_x_signals)

    x_batch, y_batch = next(generator)

    print(x_batch.shape)
    print(y_batch.shape)

    #set up validation set
    validation_data = (np.expand_dims(x_test_scaled, axis=0),
                       np.expand_dims(y_test_scaled, axis=0))


    #gru modell - keras
    model = Sequential()
    dropout = 0.1
    #LSTM modell

    model.add(LSTM(units=128,
                  return_sequences=True,
                  input_shape=(None,num_x_signals,)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
    model.add(Dropout(dropout))
    model.add(Dense(units=64))
    model.add(Dropout(dropout))
    #model.add(TimeDistributed(Dense(units=5)))



    # model.add(Dense(num_y_signals, activation='sigmoid'))

    #linear statt sigmoid


    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))

    warmup_steps = 500

    #optimizer
    optimizer = RMSprop(lr=1e-3)
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)

    print(model.summary())


    #tensorboard setup
    path_checkpoint = 'checkpoint.keras'
    #validation loss abspeichern
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)

    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=5, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./logs/',
                                       histogram_freq=0,
                                       write_graph=True)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)

    callbacks = [callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard,
                 callback_reduce_lr]

    #das Modell, für trainingssession
    if fit == True:
        model.fit_generator(generator=generator,
                            epochs=1,
                            steps_per_epoch=100,
                            validation_data=validation_data,
                            callbacks=callbacks)
    #falls checkpoint exisitert wird dieser zu validierung verwendet
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)
    #ergebniss validieren
    result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                            y=np.expand_dims(y_test_scaled, axis=0))
    result_2.append(result)
    print(col)
    print("loss (test-set):", result)

print(result_2)