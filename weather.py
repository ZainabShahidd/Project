# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read the weather data
data = pd.read_csv('C:/Users/user/Desktop/weather forcasting/year_lahore_weather_data.csv')

# Display the first few rows
print(data.head())

# Fill missing values with the column mean
data.fillna(data.mean(), inplace=True)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the Date column as the index
data.set_index('Date', inplace=True)


data_monthly = data.resample('M').mean()

# Split the data into training and testing sets
X = data_monthly.drop('Temperature', axis=1)
y = data_monthly['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using Min-Max Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualize the data (e.g., temperature)
plt.figure(figsize=(12,6))
plt.plot(data_monthly.index, data_monthly['Temperature'])
plt.title('Monthly Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()
