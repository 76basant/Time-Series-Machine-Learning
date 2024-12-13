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

