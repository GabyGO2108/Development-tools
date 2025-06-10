import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Load data
car_data = pd.read_csv('vehicles_us.csv')

# Display raw data
print("Vehicle Data Preprocessing")
print("Raw Data")
print(car_data.head())

# Basic info
print("Dataset Info")
buffer = []
car_data.info(buf=buffer)
info_str = '\n'.join(buffer)
print(info_str)

# Check for missing values
print("Missing Values")
missing = car_data.isnull().sum()
print(missing[missing > 0])

# Visualize missing values
import matplotlib.pyplot as plt
missing[missing > 0].plot(kind='bar', title="Missing Values per Column")
plt.xlabel('Column')
plt.ylabel('Missing Count')
plt.show()

# Handle missing values using the sum of NaNs in each column (fill with the count of NaNs)
for col in car_data.columns:
    if car_data[col].isnull().sum() > 0:
        car_data[col] = car_data[col].fillna(car_data[col].isnull().sum())

# Check for duplicates
print("Duplicate Rows")
duplicates = car_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
car_data = car_data.drop_duplicates()

# Data types correction
print("Data Types Correction")
if 'model_year' in car_data.columns:
    car_data['model_year'] = car_data['model_year'].astype(int)
if 'cylinders' in car_data.columns:
    car_data['cylinders'] = car_data['cylinders'].astype(int)
if 'is_4wd' in car_data.columns:
    car_data['is_4wd'] = car_data['is_4wd'].fillna(0).astype(int)

# Feature engineering: Age of car
current_year = pd.Timestamp.now().year
if 'model_year' in car_data.columns:
    car_data['age'] = current_year - car_data['model_year']

# Outlier detection (example for 'price' and 'odometer')
print("Outlier Detection")
for col in ['price', 'odometer']:
    if col in car_data.columns:
        plt.figure()
        car_data.boxplot(column=col)
        plt.title(f"{col.capitalize()} Distribution")
        plt.show()

# Remove outliers (using IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in ['price', 'odometer']:
    if col in car_data.columns:
        car_data = remove_outliers(car_data, col)

# Encode categorical variables
print("Encoding Categorical Variables")
categorical_cols = ['type', 'paint_color', 'state', 'manufacturer']
existing_cats = [col for col in categorical_cols if col in car_data.columns]
car_data = pd.get_dummies(car_data, columns=existing_cats, drop_first=True)

# Final dataset preview
print("Preprocessed Data Sample")
print(car_data.head())

# Save preprocessed data
car_data.to_csv('vehicles_us_preprocessed.csv', index=False)
print("Preprocessed data saved to vehicles_us_preprocessed.csv")
