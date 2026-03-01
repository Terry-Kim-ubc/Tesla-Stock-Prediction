import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

# --- Phase 1: Data Acquisition ---
ticker = 'TSLA'
print("Downloading data... please wait.")
# We use 'multi_level_index=False' to keep the table simple
data = yf.download(ticker, start="2020-01-01", end="2023-12-31", multi_level_index=False)

# Flatten the columns just in case
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# We only need the 'Close' prices as a simple series of numbers
series = data['Close'].dropna()

# Set the frequency to 'B' (Business days) to fix the ValueWarning
series.index = pd.DatetimeIndex(series.index).to_period('B')

# --- Phase 2: ARIMA Modeling ---
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

print("Training ARIMA model... this will take about 30 seconds.")
history = [x for x in train]
predictions = []

# The Rolling Forecast Loop
for t in range(len(test)):
    # We use a simple version of history to ensure it's just numbers
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    
    # Add the actual observation to history for the next day's prediction
    obs = test.iloc[t]
    history.append(obs)

# Calculate Accuracy (RMSE)
rmse_arima = np.sqrt(mean_squared_error(test, predictions))
print(f'\nARIMA Model RMSE: {rmse_arima:.2f}')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(test.index.to_timestamp(), test.values, label='Actual Price', color='blue')
plt.plot(test.index.to_timestamp(), predictions, label='ARIMA Prediction', color='red', linestyle='--')
plt.title(f'Tesla Stock Price Prediction - ARIMA (RMSE: {rmse_arima:.2f})')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# --- Phase 3: LSTM Preprocessing ---
from sklearn.preprocessing import MinMaxScaler

# 1. Convert data to a 2D array
# LSTM expects data in the format [Number of samples, Number of features].
# Since the current series is a 1D array, we reshape it into a 2D column array.
dataset = series.values.reshape(-1, 1)

# 2. Compress between 0 and 1 (Scaling)
# Stock prices (like $200) are too large for deep learning models to process efficiently.
# We scale all numbers so the minimum value is 0 and the maximum is 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 3. Split into training and test data (for LSTM)
# Just like ARIMA, we set 80% of the data for training.
train_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_data_len, :]

# 4. Create windows (Windowing)
# LSTM learns by looking at the past 60 days to predict the next day.
# x_train will contain the past 60 days of data, and y_train will contain the target value (the 61st day).
x_train = []
y_train = []

# Iterate from the 60th day to the last day, slicing into 60-day windows.
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0]) # Data from day 0 to 59
    y_train.append(train_data[i, 0])      # The actual price on the 60th day (Target)

# 5. Convert to Numpy arrays
# Deep learning models (TensorFlow) require 'Numpy arrays' instead of standard Python lists.
x_train, y_train = np.array(x_train), np.array(y_train)

# 6. Reshape into 3D array (Required for LSTM)
# LSTM requires a 3D data structure: [Number of samples, Time steps, Number of features].
# This tells the model: "We have [x_train length] samples, each with [60] time steps, and [1] feature (the price)."
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# --- Phase 3-2: Building the LSTM Model ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Initialize the model structure
# Sequential means we will stack layers sequentially.
model = Sequential()

# 2. Add the first LSTM layer
# Think of units (50) as the number of neurons.
# return_sequences=True means the sequence will be passed on to the next layer.
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# 3. Add the second LSTM layer
# Since this is the last LSTM layer, it won't return sequences, only the final output.
model.add(LSTM(units=50, return_sequences=False))

# 4. Output Layer (Generating the result)
# We use 1 neuron because we need a single predicted stock price.
model.add(Dense(units=1))

# 5. Compile the model (Set learning method)
# The 'adam' optimizer is an efficient algorithm for minimizing error.
# loss='mean_squared_error' measures how far off the predictions are from the actual values.
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Start Training
# epochs=10 means the model will iterate over the entire dataset 10 times.
# batch_size=32 means the model updates its weights after processing 32 samples at a time.
print("Starting LSTM training... This will take longer than ARIMA.")
model.fit(x_train, y_train, batch_size=32, epochs=10)

# --- Phase 4: Prediction and Comparison ---

# 1. Prepare test data
# Just like during training, we group the test data into 'past 60 days' windows.
test_inputs = scaled_data[len(scaled_data) - len(test) - 60:]
x_test = []
for i in range(60, len(test_inputs)):
    x_test.append(test_inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. Make predictions with the AI
# We pass the test inputs (x_test) to the trained model to get its predictions.
lstm_predictions = model.predict(x_test)

# 3. Un-scale the data (Inverse Transform)
# We revert the 0~1 scaled numbers back to their actual Dollar ($) prices.
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# 4. Calculate LSTM score (RMSE)
rmse_lstm = np.sqrt(mean_squared_error(test.values, lstm_predictions))
print(f'\nLSTM Model RMSE: {rmse_lstm:.2f}')
print(f'ARIMA Model RMSE: {rmse_arima:.2f}')

# 5. Plot the final comparison graph
plt.figure(figsize=(12, 6))
plt.plot(test.index.to_timestamp(), test.values, label='Actual Price', color='blue')
plt.plot(test.index.to_timestamp(), predictions, label='ARIMA Prediction', color='red', linestyle='--')
plt.plot(test.index.to_timestamp(), lstm_predictions, label='LSTM Prediction', color='green', linestyle='-.')
plt.title('Tesla Stock Price: ARIMA vs LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()