# Dieser code ist ein einfaches netz ohne die schleife über alle variablen des aktorendatensatzes
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
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

#def batch size
batch_size = 64
sequence_length = 1000

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



##data setup

df = read_data(nrows=20000)
print(df.head(9))
df["Widerstand"] = df["Spannung"] / df["Strom I"]
df["Abstand"]  = abs(df["Abstand"] - 4)
#filter df
target_var = ["Abstand"]
used_var = ["Zeit", "Abstand", "Widerstand"]
target_names = target_var
df = df[used_var]
#steps to predict
shift_steps = 10000
df_target = df[target_var].shift(-shift_steps)

##neural network data setup
#data
x_data = df.values[0:-shift_steps]
#target
y_data = df_target.values[:-shift_steps]
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


generator = batch_generator(batch_size, sequence_length, x_train_scaled, y_train_scaled, num_train, num_y_signals, num_x_signals)

x_batch, y_batch = next(generator)

print(x_batch.shape)
print(y_batch.shape)

#set up validation set
validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


#gru modell - keras
model = Sequential()

model.add(GRU(units=256,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

# model.add(Dense(num_y_signals, activation='sigmoid'))

#linear instaed of sigmoid
from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
init = RandomUniform(minval=-0.05, maxval=0.05)

model.add(Dense(num_y_signals,
                activation='linear',
                kernel_initializer=init))

warmup_steps = 50

#optimizer
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)

print(model.summary())


#tensorboard setup
path_checkpoint = 'checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


model.fit_generator(generator=generator,
                    epochs=50,
                    steps_per_epoch=70,
                    validation_data=validation_data,
                    callbacks=callbacks)

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)

def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


plot_comparison(start_idx=0, length=1000, train=True)
