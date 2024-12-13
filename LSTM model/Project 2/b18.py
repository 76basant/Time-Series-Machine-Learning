#Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

################################
#Data Processing
# Define the input file path
file_path = r'E:\Figures\Original Data of Aa collected by Excel & Python & Matlab of Geomagnetic Indices\Machine Laearning\Forecasting Projects\Ionspheric Forecasting\OMNIWEB_DATA.dat'


# Loading data as .dat file from this link:
#https://omniweb.gsfc.nasa.gov/html/polarity/polarity_tab.html
# read data as table data and using all columns
omni= pd.read_table(file_path, sep="\s+",  usecols=[0,1,2,3,4,5,6,7,8,9]  )
#printing data
print (omni)



omni['BZ (nT)'] = omni['1']
omni['SW Proton Density (N/cm^3)'] = omni['2']
omni['SW Plasma Speed (km/s)']=omni['3']
omni['Kp (nT)'] = omni['4']
omni['R (Sunspot No.)'] = omni['5']
omni['Dst (nT)'] = omni['6']
omni['f10.7'] = omni['7']

omni.drop(['1','2','3','4','5','6','7'],axis=1,inplace=True)

print(omni)



##################################
#searching for nan values
# Replacing specific values (999.9, 9999999., 99.99) with NaN
omni.replace([999.9, 9999999., 99.99], np.nan, inplace=True)

# Printing the updated DataFrame after replacing values
print("\nData after replacing specific values with NaN:")
print(omni)

# Counting NaN values in each column
nan_count = omni.isna().sum()

# Printing the number of NaN values for each column
print("\nNumber of NaN values in each column:")
print(nan_count)

total_count = len(omni)        # Total values in each column
known_count = total_count - nan_count  # Non-NaN values


# Counting total NaN values across all columns
total_nan = omni.isna().sum().sum()  # Total NaN values across all columns
total_values = omni.size  # Total values (rows * columns)
total_known = total_values - total_nan  # Known values (non-NaN)

# Plotting a single pie chart for NaN vs Known values
labels = ['Known Values', 'NaN Values']
sizes = [total_known, total_nan]  # Values for the pie chart
explode = (0, 0.1)  # Explode the NaN section for emphasis

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title('NaN vs Known Values Across All Columns')
plt.show()

##########################
#we have multiple methods to fill Nan values 

#first step
# Applying linear interpolation to fill missing values
#omni.interpolate(method='linear', inplace=True)
#Filling NaN values with mean values
# omni.fillna(omni.mean(), inplace=True)

#second step 
# Filling NaN values with the last valid value
#omni.fillna(method='ffill', inplace=True)

#third step 
# Filling NaN values with the next valid value
#omni.fillna(method='bfill', inplace=True)

#fourth step
# Filling NaN values first with forward fill, then backward fill
omni.fillna(method='ffill', inplace=True)
omni.fillna(method='bfill', inplace=True)

# Printing the DataFrame after interpolation
print("after filling")
print(omni.iloc[:,:4])

omni['datetime'] = pd.to_datetime(omni['YEAR'].astype(str) + ' ' + omni['DOY'].astype(str) + ' ' + omni['HR'].astype(str), format='%Y %j %H')

omni['datetime'] = pd.to_datetime(omni['datetime'])

# Correcting the slicing error
print(omni.iloc[:, 3:10])

######################
#correlation matrix
df = pd.DataFrame(omni.iloc[:, 3:10])

# Calculate the correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Create a heatmap
plt.figure(figsize=(8, 6))  # Adjust the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# Add title and show plot
plt.title("Heatmap of Correlation Coefficients")
plt.show()


###################################
#code of Deep Learning
df=omni
# Feature selection with domain knowledge
# (Replace with your domain-specific feature selection logic)
selected_features = ['YEAR', 'DOY', 'HR','Kp (nT)','R (Sunspot No.)','f10.7']  # Replace with relevant features
X = df[selected_features]
Y = df['BZ (nT)']

print("Independent Variable")
print(X)

print("Dependent Variable")
print(Y)
# Split data into train and test sets (consider stratified splitting for imbalanced data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, shuffle=True)

# Normalize data using robust scaler for potential outliers
scaler_X = MinMaxScaler(feature_range=(0, 1))  # Adjust range for specific use cases
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1))
Y_test_scaled = scaler_Y.transform(Y_test.values.reshape(-1, 1))

# Apply PCA with explained variance ratio determination
pca = PCA(n_components=0.95)  # Adjust based on desired information retention
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Number of features after PCA
features_pca = X_train_pca.shape[1]

look_back = 10  # Adjust look_back based on data patterns and experiment

def create_sequences(data, look_back):
    sequences = [data[i:i + look_back] for i in range(len(data) - look_back)]
    return np.array(sequences)

# Create sequences
X_train_sequences = create_sequences(X_train_pca, look_back)
X_test_sequences = create_sequences(X_test_pca, look_back)

# Create target sequences
Y_train_sequences = create_sequences(Y_train_scaled, look_back)
Y_test_sequences = create_sequences(Y_test_scaled, look_back)

# Adjust target sequences to match input sequences (consider multi-step forecasting)
Y_train_sequences = Y_train_sequences[:, -1, 0]  # Last value in the sequence
Y_test_sequences = Y_test_sequences[:, -1, 0]   # Last value in the sequence

###############

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example data (replace with actual data)
look_back = 10  # Number of previous time steps
features_pca = 5  # Number of features after PCA transformation
X_train_sequences = np.random.rand(100, look_back, features_pca)  # 100 samples, look_back time steps, features_pca features
Y_train_sequences = np.random.rand(100, 1)  # 100 target values
X_test_sequences = np.random.rand(20, look_back, features_pca)  # 20 test samples
Y_test_sequences = np.random.rand(20, 1)  # 20 test targets

# ModelCheckpoint callback
checkpoint4_path = "model4_weights.keras"

checkpoint_callback4 = ModelCheckpoint(
    filepath=checkpoint4_path,
    monitor='val_loss',  # or 'val_accuracy', depending on what you want to monitor
    save_best_only=True,  # Save only the best model weights
    mode='min',  # 'min' for loss (lower is better), 'max' for accuracy (higher is better)
    verbose=1  # Verbosity mode
)

# Build the BiLSTM model
model_bilstm = Sequential()
model_bilstm.add(Input(shape=(look_back, features_pca)))  # Explicitly define input shape
model_bilstm.add(Bidirectional(LSTM(units=50, return_sequences=True)))
model_bilstm.add(LSTM(units=25, return_sequences=False))
model_bilstm.add(Dense(1, activation='linear'))
model_bilstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model_bilstm.fit(X_train_sequences, Y_train_sequences,
                           epochs=50,
                           validation_split=0.2,
                           callbacks=[checkpoint_callback4]  # Include the callback here
)

# Load the best weights
model_bilstm.load_weights(checkpoint4_path)

# Predict on training and test data
Y_train_pred = model_bilstm.predict(X_train_sequences)
Y_test_pred = model_bilstm.predict(X_test_sequences)

# Inverse transform predictions (if applicable)
# Y_train_inv = scaler_Y.inverse_transform(Y_train_sequences.reshape(-1, 1))
# Y_train_pred_inv = scaler_Y.inverse_transform(Y_train_pred)
# Y_test_inv = scaler_Y.inverse_transform(Y_test_sequences.reshape(-1, 1))
# Y_test_pred_inv = scaler_Y.inverse_transform(Y_test_pred)

# Calculate metrics
mse_train = mean_squared_error(Y_train_sequences, Y_train_pred)
r2_train = r2_score(Y_train_sequences, Y_train_pred)
mse_test = mean_squared_error(Y_test_sequences, Y_test_pred)
r2_test = r2_score(Y_test_sequences, Y_test_pred)

# Print the results
print(f"Training Mean Squared Error: {mse_train}")
print(f"Training R² Score: {r2_train}")
print(f"Test Mean Squared Error: {mse_test}")
print(f"Test R² Score: {r2_test}")

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
