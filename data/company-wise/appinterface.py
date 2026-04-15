"""  so whats been completed till now?
        user ley stock symbol ra date range input dinxa
        tyo particular date range vitra particular stock symbol ko data fetch hunxa
        data display hunxa
        chart haru vayo
        x axis maa proper date dekhaaune ra y axis maa closing prices dekhaune vayo
        model ready vayo(linear regression)

        REMAININGS:
        x ra y label dekhaayena idk why. gotta look into that
        login page banauna baaki
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data
import streamlit as st
import datetime
from sklearn.preprocessing import MinMaxScaler
import os
#from keras.models import load_model
#from pathlib import Path

st.set_page_config(page_title="MarketLens Python", layout = 'centered' )

st.title("📈MarketLens: Stock Market Data Analysis and Prediction App")


start = st.datetime_input("Enter the initial date:", min_value= datetime.datetime(2008,12,1,18,45), max_value="now")
end = st.datetime_input("Enter the later date:", min_value= datetime.datetime(2008,12,1,18,45), max_value="now")


user_input = st.text_input("Enter Stock Ticker eg: NABIL")
input = user_input + ".csv"
st.subheader(f"Data from {start} to {end}")


# Load the CSV file
#st.write("current working directory "+ os.getcwd())
base_path = (r'D:\Sumeedi\project_final\nepse-data\data\company-wise')
df = pd.read_csv(base_path +"\\"+ input)

# Save as a comma-delimited text file
df.to_csv(f'{user_input}.txt', sep=',', index=False)

#df = np.genfromtxt("ADBL.txt", delimiter = ',' ,skip_header=1)

df["published_date"] = pd.to_datetime(df["published_date"])
df1 = df[df["published_date"].between(start,end)]
df2 = df.dropna()

st.write(df1)


#describing data
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.published_date, df.close)
st.pyplot(fig)
plt.xlabel('Time')
plt.ylabel('Closing Price')

st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(df.close)
st.pyplot(fig)
plt.xlabel('Time')
plt.ylabel('Price')

st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.close)
st.pyplot(fig)
plt.xlabel('Time')
plt.ylabel('Price')

#splitting data into Training and Testing
data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])    #[['open','high','low','close','traded_quantity']] for using all OHLCV to train
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70) : int(len(df))])    #or use .iloc as df[[]].iloc(int(len(df)*0.70): )

scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

#load the training model (this model is non-linear regression model with RELU activation function)
#model = load_model('LSTM_model')


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

data = np.array(input_data).astype(float)


x_test = []
y_test = []

for i in range(100,data.shape[0]):
    x = data[i-100:i].reshape(100,1)
    x_test.append(x)
    y_test.append(data[i,0])     #i,0 in sense close column 1st maa xa vanera, here date is 5th col in csv file

# st.write(data_training.shape)      2382,1  total
# st.write(final_df.shape)           1121,1   training
# st.write(input_data.shape)         1121,1    training
#st.write(data_testing.shape)         1021,1    testing
#st.write(x_test.shape)   list doesnot have attribute "shape"
#st.write(input_data[:5])


x_test = np.array(x_test)
y_test = np.array(y_test)

#y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
#y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# #final graph
# st.subheader('Predicted vs original')
# fig2 = plt.figure(figsize=(12,6))
# plt.plot(y_test,'b', label = 'original price')
# plt.plot(y_predicted,'r', label = 'predicted price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)


# # --- Custom Linear Regression (From Scratch) ---
class CustomLinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.n_features = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features = n_features

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)

        # FIX: ensure correct shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # safety check (prevents your exact error)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature mismatch! Model expects {self.n_features}, "
                f"but got {X.shape[1]}"
            )
        return np.dot(X, self.weights) + self.bias

 # Prediction Logic
st.subheader("Next Day Prediction")
if st.button("Predict Closing Price(using linear regression)"):
    # Training on the 100 rows in csv
    df_last_100 = df.tail(100)
    X = df_last_100[['open', 'high', 'low', 'traded_quantity']].values
    y = df_last_100['close'].values
        
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std
        
    # Train model
    model = CustomLinearRegression(lr=0.1, iterations=2000)
    model.fit(X_norm, y)
        
    # Predict using 100 days data
    last_day = X_norm[-1].reshape(1, -1)
    x_test_linear_regression = x_test.reshape(x_test.shape[0], -1)
    prediction = model.predict(x_test_linear_regression)
  

    #final graph
    st.subheader('Predicted vs original')
    fig3 = plt.figure(figsize=(12,6))
    #plt.plot(df.published_date, df.close)
    plt.plot(prediction,'b', label = 'predicted price')
    plt.plot(y_test,'r', label = 'original price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)
        
    st.metric("Predicted Close", f"Rs.{prediction[0]:.2f}")

