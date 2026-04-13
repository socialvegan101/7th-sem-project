"""  so whats been completed till now?
        user ley stock symbol ra date range input dinxa
        tyo particular date range vitra particular stock symbol ko data fetch hunxa
        data display hunxa
        chart haru ali ali vayo
        x axis maa proper date dekhaaune ra y axis maa closing prices dekhaune vayo

        REMAININGS:
        x ra y label dekhaayena idk why. gotta look into that
        model banauna baaki
        login page banauna baaki
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data
import streamlit as st
#from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
#from keras.models import load_model
#from pathlib import Path

st.set_page_config(page_title="MarketLens Python", layout = 'centered' )

st.title("📈MarketLens: Stock Market Data Analysis and Prediction App")


start = st.datetime_input("Enter the initial date:")
end = st.datetime_input("Enter the later date:")


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
data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70) : int(len(df))])

scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

#load the training model
#model = load_model('LSTM_model')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:1])
    y_test.append(input_data[i,1])     #i,0 thyo in sense close column 1st maa xa vanera, here date is 1st col

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final graph
st.subheader('Predicted vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = 'original price')
plt.plot(y_predicted,'r', label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

