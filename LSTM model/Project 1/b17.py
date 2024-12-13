import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import openpyxl
from openpyxl.workbook import Workbook
from openpyxl import load_workbook



# Define the input file path
file_path = r'E:\Figures\Original Data of Aa collected by Excel & Python & Matlab of Geomagnetic Indices\Machine Laearning\b10.dat'


# Loading data as .dat file from this link:
#https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html
# read data as table data and using all columns
df1= pd.read_table(file_path, sep="\s+",  usecols=[0,1,2]  )
#printing data
print (df1)

#if you want to rename the columns, you should use that:
df1.columns= ['Year', 'Month','SSA']
print(df1)



# Calculate monthly averages (if needed)
# For this example, it's assumed the data is already monthly
monthly_data = df1.copy()

# Find rows with NaN values
nan_rows = monthly_data[monthly_data.isnull().any(axis=1)]

# Display the rows with NaN values
print("Rows with NaN values:")
print(nan_rows)

monthly_data=monthly_data.iloc[7:]
print(monthly_data)

# Generate time as fractional years
k = 1875  # Start year
dt = 1 / 12.0  # Monthly step as a fraction of a year
time = np.arange(k, k + len(monthly_data) * dt, dt)[:len(monthly_data)]
#print(time)
# Add time to the DataFrame
monthly_data['time'] = time


print(monthly_data)

window=13
# Calculate 13-month Simple Moving Average (SMA)
monthly_data['SMA_13'] = monthly_data['SSA'].rolling(window=13).mean()
print(monthly_data.head(13))

monthly_data=monthly_data.iloc[12:]

print(monthly_data)

#monthly_data=monthly_data.iloc[1:]
#print(monthly_data)

# Print monthly data to verify
#print(monthly_data)

# Scatter plot of monthly data
plt.figure(figsize=(10, 6))
plt.scatter(monthly_data['time'], monthly_data['SMA_13'], label='Monthly Data', marker='o', color='red',s=10)

# Cubic interpolation
cubic_interpolation_model = interp1d(monthly_data['time'], monthly_data['SMA_13'], kind="cubic")

# Generate interpolated values for the full range of time data
X1_ = np.linspace(monthly_data['time'].min(), monthly_data['time'].max(), 1000)  # Dense range for smooth curve
Y2_ = cubic_interpolation_model(X1_)

# Plot interpolated curve
plt.plot(X1_, Y2_, color='red', linestyle=':', label='Cubic Interpolation')

# Plot formatting
plt.xlabel('Year')
plt.ylabel('SSA Monthly Average')
plt.title('Monthly Average of SSA with Interpolation')
plt.legend()
plt.grid(True)
plt.show()






from sklearn.linear_model import LinearRegression

# Split the data sequentially (60% training, 40% testing)
train_size = int(0.6 * len(monthly_data))
x_train = monthly_data[['time']].values[:train_size]
y_train = monthly_data['SMA_13'].values[:train_size]
x_test = monthly_data[['time']].values[train_size:]
y_test = monthly_data['SMA_13'].values[train_size:]
#print(x_train)
# Linear regression model for trend fitting
model_trend = LinearRegression()
model_trend.fit(x_train, y_train)

# Predict fitted and forecasted values
y_fittedvalue = model_trend.predict(x_train)
y_forecast = model_trend.predict(x_test)

# Plot training data and trend line

plt.scatter(x_train, y_train, color='blue', label='Training Data', s=10)
#plt.plot(x_train, y_train, color='blue')
plt.plot(x_train, y_fittedvalue, color='green', label='Fitted Trend')

# Plot test data and forecast
plt.scatter(x_test, y_test, color='orange', label='Test Data', s=10)
#plt.plot(x_test, y_test, color='orange')
plt.plot(x_test, y_forecast, color='red', label='Forecast')

plt.xlabel('Time')
plt.ylabel('SSA monthly Average')
plt.title('SSA monthly Average - Trend and Forecast')
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate date range
dates = monthly_data['time']  # Monthly frequency
print(len(dates))
print(len(time))


# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data[['SMA_13']])

# Prepare data for LSTM
def create_sequences(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

time_steps = 5
length_split = 0.3
x, y = create_sequences(scaled_data, time_steps)

# Split data
x_train, y_train = x[:int(length_split * len(x))], y[:int(length_split * len(y))]
x_test, y_test = x[int(length_split * len(x)):], y[int(length_split * len(y)):]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)


# Predict and inverse transform predictions
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

# Create a corresponding date range for the test predictions
test_dates = dates[-len(y_test):]
print(test_dates)
# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(dates, monthly_data['SMA_13'], label='Historical Data', color='blue',s=10)
plt.scatter(test_dates, y_pred, label='Forecast', color='red',s=5)
plt.xlabel('Date')
plt.ylabel('SSA monthly Average')
plt.title('LSTM Forecast for SSA')
plt.legend()
plt.grid(True)
plt.show()



