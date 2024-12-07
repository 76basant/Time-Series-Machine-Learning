
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the input file path
file_path = r'E:\Figures\Original Data of Aa collected by Excel & Python & Matlab of Geomagnetic Indices\Machine Laearning\HSSA.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Calculate monthly averages (if needed)
# For this example, it's assumed the data is already monthly
monthly_data = df.copy()

# Find rows with NaN values
nan_rows = monthly_data[monthly_data.isnull().any(axis=1)]

# Display the rows with NaN values
print("Rows with NaN values:")
print(nan_rows)

monthly_data.rename(columns={'SSA_N': 'SSA_N_monthly_Avg'}, inplace=True)


# Generate time as fractional years
k = 1986  # Start year
dt = 1 / 12.0  # Monthly step as a fraction of a year
time = np.arange(k, k + len(monthly_data) * dt, dt)[:len(monthly_data)]
#print(time)
# Add time to the DataFrame
monthly_data['time'] = time

# Print monthly data to verify
#print(monthly_data)

# Scatter plot of monthly data
plt.figure(figsize=(10, 6))
plt.scatter(time, monthly_data['SSA_N_monthly_Avg'], label='Monthly Data', marker='o', color='red',s=10)

# Cubic interpolation
cubic_interpolation_model = interp1d(time, monthly_data['SSA_N_monthly_Avg'], kind="cubic")

# Generate interpolated values for the full range of time data
X1_ = np.linspace(time.min(), time.max(), 1000)  # Dense range for smooth curve
Y2_ = cubic_interpolation_model(X1_)

# Plot interpolated curve
#plt.plot(X1_, Y2_, color='red', linestyle=':', label='Cubic Interpolation')

# Plot formatting
plt.xlabel('Year')
plt.ylabel('SSA_N Monthly Average')
plt.title('Monthly Average of SSA_N with Interpolation')
plt.legend()
plt.grid(True)
plt.show()



from sklearn.linear_model import LinearRegression

# Split the data sequentially (60% training, 40% testing)
train_size = int(0.6 * len(monthly_data))
x_train = monthly_data[['time']].values[:train_size]
y_train = monthly_data['SSA_N_monthly_Avg'].values[:train_size]
x_test = monthly_data[['time']].values[train_size:]
y_test = monthly_data['SSA_N_monthly_Avg'].values[train_size:]
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
plt.ylabel('SSA_N monthly Average')
plt.title('SSA_N monthly Average - Trend and Forecast')
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
dates = pd.date_range(start="1986-01", end="2022-01", freq="M")  # Monthly frequency

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data[['SSA_N_monthly_Avg']])

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
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)


# Predict and inverse transform predictions
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

# Create a corresponding date range for the test predictions
test_dates = dates[-len(y_test):]
print(test_dates)
# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(dates, monthly_data['SSA_N_monthly_Avg'], label='Historical Data', color='blue',s=15)
plt.plot(test_dates, y_pred, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('SSA_N monthly Average')
plt.title('LSTM Forecast for SSA_N')
plt.legend()
plt.grid(True)
plt.show()
