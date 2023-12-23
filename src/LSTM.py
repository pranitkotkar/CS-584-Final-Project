#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:27:08 2023

@author: amaterasu
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt



dataset = pd.read_csv('data/household_power_consumption_daily.csv', parse_dates=['datetime'], index_col=['datetime'])


train_data = dataset.values[1:-328]
test_data = dataset.values[-328:-6]

# Restructure into windows of weekly data
train = np.array(np.split(train_data, len(train_data) / 7))
test = np.array(np.split(test_data, len(test_data) / 7))


print(f"Train shape: {train.shape}")
print(f"First value: {train[0, 0, 0]}, Last value: {train[-1, -1, 0]}")


print(f"Test shape: {test.shape}")
print(f"First value: {test[0, 0, 0]}, Last value: {test[-1, -1, 0]}")

class LSTMEncoderDecoder:
    def __init__(self, n_input, n_out=7, epochs=50, batch_size=16):
        self.n_input = n_input
        self.n_out = n_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def to_supervised(self, train):
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = [], []
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.n_input
            out_end = in_end + self.n_out
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            in_start += 1
        return np.array(X), np.array(y)

    def build_model(self, train):
        train_x, train_y = self.to_supervised(train)
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        self.model.add(tf.keras.layers.RepeatVector(n_outputs))
        self.model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
        self.model.compile(loss='mse', optimizer='adam')

        history = self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_split=0.1)
        return history

    def forecast(self, history):
        data = np.array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        input_x = data[-self.n_input:, :]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        yhat = self.model.predict(input_x, verbose=0)
        return yhat[0]


    
    def evaluate_forecasts(self, actual, predicted):
        rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores = [], [], [], [], []
        if actual.shape != predicted.shape:
            raise ValueError(f"Shape mismatch: Actual shape {actual.shape}, Predicted shape {predicted.shape}")

        for i in range(actual.shape[1]):
            mse = self.calculate_mse(actual[:, i], predicted[:, i])
            mae = self.calculate_mae(actual[:, i], predicted[:, i])
            mape = self.calculate_mape(actual[:, i], predicted[:, i])
            smape = self.calculate_smape(actual[:, i], predicted[:, i])
            rmse = math.sqrt(mse)

            mse_scores.append(mse)
            mae_scores.append(mae)
            mape_scores.append(mape)
            smape_scores.append(smape)
            rmse_scores.append(rmse)

        return rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores


        
    def summarize_scores(self, name, rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores):
        print(f"{name} RMSE: {', '.join([f'{s:.1f}' for s in rmse_scores])}")
        print(f"{name} MSE: {', '.join([f'{s:.1f}' for s in mse_scores])}")
        print(f"{name} MAPE: {', '.join([f'{s:.1f}' for s in mape_scores])}")
        print(f"{name} SMAPE: {', '.join([f'{s:.1f}' for s in smape_scores])}")
        print(f"{name} MAE: {', '.join([f'{s:.1f}' for s in mae_scores])}")


    
    def evaluate_model(self, train, test):
        history = self.build_model(train)
        predictions = []  
        for i in range(len(test)):
            yhat_sequence = self.forecast(train)
            
            yhat_sequence = yhat_sequence.reshape((test.shape[1],))
            predictions.append(yhat_sequence)  
           
            train = np.append(train, [test[i, :]], axis=0)
            train = train.reshape((-1, test.shape[1], test.shape[2]))


        predictions = np.array(predictions)
            
            
        if predictions.shape != test[:, :, 0].shape:
            raise ValueError(f"Shape mismatch in predictions. Expected shape: {test[:, :, 0].shape}, but got: {predictions.shape}")
                
        rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores = self.evaluate_forecasts(test[:, :, 0], predictions)
        return history, rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores, predictions
            
    
    def plot_rmse_scores(self, rmse_scores, label='LSTM'):
        days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        plt.figure(figsize=(10, 5))
        plt.plot(days, rmse_scores, marker='o', label=label)
        plt.title('Root Mean Squared Error by Day')
        plt.ylabel('Root Mean Squared Error')
        plt.xlabel('Day of Week')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_training_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Training vs Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
    def plot_actual_vs_predicted(self, test, predictions, week):
        actual = test[week, :, 0]
        predicted = predictions[week, :] 

        plt.figure(figsize=(10, 5))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.title(f'Actual vs Predicted Power Consumption - Week {week+1}')
        plt.ylabel('Power Consumption')
        plt.xlabel('Day')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def calculate_mae(self, actual, predicted):
        return mean_absolute_error(actual, predicted)

    def calculate_mse(self, actual, predicted):
        return mean_squared_error(actual, predicted)

    def calculate_mape(self, actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    def calculate_smape(self, actual, predicted):
        return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


model = LSTMEncoderDecoder(n_input=14)
history, rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores, predictions = model.evaluate_model(train, test)


model.summarize_scores('lstm', rmse_scores, mse_scores, mape_scores, smape_scores, mae_scores)

# Plot training vs validation loss
model.plot_training_loss(history)

# Plot actual vs predicted for a specific week (e.g., first week in the test set)
model.plot_actual_vs_predicted(test, predictions, week=0)

# Plot RMSE scores
model.plot_rmse_scores(rmse_scores)
plt.show()