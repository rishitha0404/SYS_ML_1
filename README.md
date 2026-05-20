# SYS_ML_1
Perform an initial assessment of the dataset to understand its structure, size, and quality.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("train_data.csv")

print("Dataset Loaded Successfully")
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns)
train_table = df[['Train_Name', 'Source_Station', 'Destination_Station']]

print("\nTrain Wise Table:")
print(train_table.head())
print("\nDistance Statistics")
print(df['Distance'].describe())

print("\nStops Statistics")
print(df['Stops'].describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
