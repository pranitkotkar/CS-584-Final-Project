#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:12:03 2023

@author: amaterasu
"""


import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot
import tensorflow as tf
from pandas import read_csv


def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_smape(actual, predicted):
    return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


# split multivariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train)/7))
    test = np.array(np.split(test, len(test)/7))
    return train, test

def evaluate_forecasts(actual, predicted):
    scores_mse = []
    scores_rmse = []
    scores_mae = []
    scores_mape = []
    scores_smape = []

    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        scores_mse.append(mse)

        rmse = sqrt(mse)
        scores_rmse.append(rmse)

        mae = mean_absolute_error(actual[:, i], predicted[:, i])
        scores_mae.append(mae)

        mape = calculate_mape(actual[:, i], predicted[:, i])
        scores_mape.append(mape)

        smape = calculate_smape(actual[:, i], predicted[:, i])
        scores_smape.append(smape)

    return scores_mse, scores_rmse, scores_mae, scores_mape, scores_smape


def summarize_scores(name, scores_mse, scores_rmse, scores_mae, scores_mape, scores_smape):
    print("MSE: ", ', '.join([f"{score:.2f}" for score in scores_mse]))
    print("RMSE: ", ', '.join([f"{score:.2f}" for score in scores_rmse]))
    print("MAE: ", ', '.join([f"{score:.2f}" for score in scores_mae]))
    print("MAPE: ", ', '.join([f"{score:.2f}" for score in scores_mape]))
    print("SMAPE: ", ', '.join([f"{score:.2f}" for score in scores_smape]))

def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])   
        in_start += 1
    return np.array(X), np.array(y)


def build_model(train, n_steps, n_length, n_input):
    
    train_x, train_y = to_supervised(train, n_input)
    
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape into subsequences [samples, time steps, rows, cols, channels]
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    model = tf.keras.models.Sequential([
        tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.RepeatVector(n_outputs),
        tf.keras.layers.LSTM(200, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
    ])
    model.compile(loss='mse', optimizer='adam')
    # fit model and return history
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1)
    return model, history


def forecast(model, history, n_steps, n_length, n_input, n_features):
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    input_x = input_x.reshape((1, n_steps, 1, n_length, n_features))
    yhat = model.predict(input_x, verbose=0)
    yhat = yhat[0]
    return yhat

# evaluate a single model
def evaluate_model(train, test, n_steps, n_length, n_input, n_features):
    n_features = 8
    model, fit_history = build_model(train, n_steps, n_length, n_input)
    
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_steps, n_length, n_input, n_features)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = np.array(predictions)
    scores_mse, scores_rmse, scores_mae, scores_mape, scores_smape = evaluate_forecasts(test[:, :, 0], predictions)
    return model, fit_history, predictions, scores_mse, scores_rmse, scores_mae, scores_mape, scores_smape


dataset = read_csv('data/household_power_consumption_daily.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

train, test = split_dataset(dataset.values)

# number of steps and the length of steps
n_steps, n_length = 2, 7
# total days to use as input
n_input = n_length * n_steps

n_features = dataset.shape[1]  # n_features

model, fit_history, predictions_plot, scores_mse, scores_rmse, scores_mae, scores_mape, scores_smape = evaluate_model(train, test, n_steps, n_length, n_input, n_features)
summarize_scores('Conv-LSTM', scores_mse, scores_rmse, scores_mae, scores_mape, scores_smape)

# Plot actual vs predicted values for a sample week
sample_week_index = 0  # example, the first week in the test set
actuals = test[:, :, 0]
predictions_plot = np.array(predictions_plot)

pyplot.figure()
pyplot.plot(actuals[sample_week_index], label='Actual')
pyplot.plot(predictions_plot[sample_week_index], label='Predicted')
pyplot.title(f'Actual vs Predicted - Week {sample_week_index+1}')
pyplot.legend()
pyplot.show()

# Plot training vs validation loss
pyplot.figure()
pyplot.plot(fit_history.history['loss'], label='Training Loss')
if 'val_loss' in fit_history.history:
    pyplot.plot(fit_history.history['val_loss'], label='Validation Loss')
pyplot.title('Training vs Validation Loss')
pyplot.legend()
pyplot.show()

# plot rmse scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.title('Day-wise RMSE Scores')
pyplot.plot(days, scores_rmse, marker='o', label='lstm')
pyplot.show()