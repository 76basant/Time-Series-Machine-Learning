# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import os
from google.colab import drive
drive.mount('/content/drive')

# Change to your desired folder
os.chdir('/content/drive/My Drive/Machine Learning Codes/Time Series Machine Learning/Forecasting Models/')

# Loading data as .dat file from this link:
# https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html
# Read data as table data and using all columns
omni = pd.read_table('OMNIWEB_DATA.dat', sep="\s+",  usecols=[0,1,2,3,4,5,6,7,8,9])
#printing data
print(omni)

# Data processing
omni['BZ (nT)'] = omni['1']
omni['SW Proton Density (N/cm^3)'] = omni['2']
omni['SW Plasma Speed (km/s)'] = omni['3']
omni['Kp (nT)'] = omni['4']
omni['R (Sunspot No.)'] = omni['5']
omni['Dst (nT)'] = omni['6']
omni['f10.7'] = omni['7']

omni.drop(['1','2','3','4','5','6','7'], axis=1, inplace=True)
print(omni)

# Replacing specific values (999.9, 9999999., 99.99) with NaN
omni.replace([999.9, 9999999., 99.99, 9999.], np.nan, inplace=True)
print("\nData after replacing specific values with NaN:")
print(omni)

# Filling NaN values with forward fill and then backward fill
omni.fillna(method='ffill', inplace=True)
omni.fillna(method='bfill', inplace=True)

# Printing the DataFrame after filling NaN values
print("After filling missing values:")
print(omni.iloc[:,:4])

# Convert YEAR, DOY, HR to datetime
omni['datetime'] = pd.to_datetime(omni['YEAR'].astype(str) + ' ' + omni['DOY'].astype(str) + ' ' + omni['HR'].astype(str), format='%Y %j %H')
print(omni.iloc[:, 3:10])

# Correlation matrix
df = pd.DataFrame(omni.iloc[:, 3:10])
correlation_matrix = df.corr()
print(correlation_matrix)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Correlation Coefficients")
plt.show()

# Feature selection and data preprocessing
df = omni.iloc[:,[7,9]]# Replace with your dataset
print("all data before applying deep learning")
print(df)
selected_features_X = ['R (Sunspot No.)']  # Check correlation first
X = df[selected_features_X]
selected_features_Y = ['f10.7']
Y = df[selected_features_Y]

# Check correlation to ensure features are relevant
correlation_matrix = df.corr()
print("Correlation with target variable:", correlation_matrix['f10.7'])

# Define the length of the split (e.g., 80% for training and 20% for testing)
length_split = 0.8  # Adjusted for better training set size

# Split the data into training and test sets
X_train, Y_train = X[:int(length_split * len(X))], Y[:int(length_split * len(Y))]
X_test, Y_test = X[int(length_split * len(X)):], Y[int(length_split * len(Y)):]

# Normalize data using MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_scaled = scaler_Y.transform(Y_test.values.reshape(-1, 1))

# Sequence creation for LSTM (ensure look-back is set properly for time series)
look_back = 10  # Number of previous time steps to consider

def create_sequences(data, look_back):
    sequences = []
    for i in range(len(data) - look_back):
        sequences.append(data[i:i + look_back])
    return np.array(sequences)

X_train_sequences = create_sequences(X_train_scaled, look_back)
X_test_sequences = create_sequences(X_test_scaled, look_back)
Y_train_sequences = create_sequences(Y_train_scaled, look_back)
Y_test_sequences = create_sequences(Y_test_scaled, look_back)

# Adjust target sequences to match input sequences
Y_train_sequences = Y_train_sequences[:, -1, 0]  # Last value in the sequence
Y_test_sequences = Y_test_sequences[:, -1, 0]  # Last value in the sequence

# LSTM model definition (adjusted architecture with Dropout)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_sequences.shape[1], X_train_sequences.shape[2])))
model.add(Dropout(0.3))  # Increased dropout rate to reduce overfitting
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.3))  # Increased dropout rate
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_sequences, Y_train_sequences, epochs=50, batch_size=32, validation_data=(X_test_sequences, Y_test_sequences))

# Evaluate the model
loss = model.evaluate(X_test_sequences, Y_test_sequences)
print(f"Test Loss: {loss}")

# Predictions
predictions = model.predict(X_test_sequences)

# Inverse transform predictions and actual values for comparison
predictions_inverse = scaler_Y.inverse_transform(predictions)
Y_test_actual = scaler_Y.inverse_transform(Y_test_scaled[look_back:])

# Display results
#print("Predictions:", predictions_inverse.flatten())
#print("Actual Values:", Y_test_actual.flatten())

# Evaluate performance with additional metrics
mse = mean_squared_error(Y_test_actual, predictions_inverse)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_actual, predictions_inverse)
r2 = r2_score(Y_test_actual, predictions_inverse)

# Print metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")





# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test_actual.flatten(), predictions_inverse.flatten(), label='Predictions vs Actual',s=1)
plt.legend()
plt.title('Actual vs Predicted f10.7')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()



# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(Y_test_actual, label='Actual Values')
plt.plot(predictions_inverse, label='Predictions')
plt.legend()
plt.title('Actual vs Predicted f10.7')
plt.xlabel('Time')
plt.ylabel('f10.7')
plt.show()



# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(Y_test_actual)), Y_test_actual, label='Actual Values', s=1)
plt.scatter(np.arange(len(predictions_inverse)), predictions_inverse, label='Predictions', s=1)
plt.legend()
plt.title('Actual vs Predicted f10.7')
plt.xlabel('Time')
plt.ylabel('f10.7')
plt.show()
