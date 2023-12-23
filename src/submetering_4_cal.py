# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np


dataset = pd.read_csv('data/household_power_consumption.txt', sep=';', parse_dates={'datetime': [0, 1]}, index_col=['datetime'], low_memory=False, infer_datetime_format=True)

# Replace '?' with NaN
dataset.replace('?', np.nan, inplace=True)
dataset = dataset.astype('float32')

#fill missing values
def fill_missing(data):
    per_day = 60 * 24
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pd.isna(data.iloc[i, j]):
                data.iloc[i, j] = data.iloc[i - per_day, j]

fill_missing(dataset)


dataset['sub_metering_4'] = (dataset.iloc[:, 0] * 1000 / 60) - dataset.iloc[:, 4:7].sum(axis=1)

# Resample minute data to total for each day and sum
daily_sum = dataset.resample('D').sum()
daily_sum.to_csv('data/household_power_consumption_daily.csv')


print(daily_sum.shape)
print(daily_sum.head())
