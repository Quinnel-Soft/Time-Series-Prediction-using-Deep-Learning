!pip install pandas numpy matplotlib tensorflow scikit-learn

from google.colab import files
uploaded = files.upload()


# Importing the libraries that required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Loading the dataset from uploaded files
df = pd.read_csv('retail_sales.csv')

# Display the column names
print("Column names in the dataset:")
print(df.columns)

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Ploting the sales data
df['Sales'].plot(figsize=(16, 8))
plt.title('Retail Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train['Sales'].values, seq_length)
X_test, y_test = create_sequences(test['Sales'].values, seq_length)

#LSTM

# Training data for LSTM.
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Defining the model
model = Sequential([
    LSTM(10, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(10, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Make predictions
predictions = model.predict(X_test)

#Evaluation
# Metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))  # Root Mean Square Error (RMSE)
mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error (MAE)
r2 = r2_score(y_test, predictions)  # Coefficient of Determination (R2) 

# printing the output
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

# Plot the results
plt.figure(figsize=(18, 5))
plt.plot(df.index[train_size+seq_length:], y_test, label='Actual Sales')
plt.plot(df.index[train_size+seq_length:], predictions, label='Predicted Sales')
plt.title('Retail Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Plot the actual sales
plt.figure(figsize=(14, 6))
plt.plot(df.index[train_size+seq_length:], y_test, label='Actual Sales', color='blue')
plt.title('Actual Retail Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Plot the predicted sales
plt.figure(figsize=(14, 6))
plt.plot(df.index[train_size+seq_length:], predictions, label='Predicted Sales', color='red')
plt.title('Predicted Retail Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
