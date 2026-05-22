import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import regressionModel as rm
import streamlit as st
import datetime
from sklearn.preprocessing import MinMaxScaler
#import plotly.graph_objects as go
import os
#from keras.models import load_model
#from pathlib import Path

st.set_page_config(page_title="MarketLens Python", layout = 'centered' )

st.title("📈MarketLens: Stock Data Analysis and Prediction App")


# History Section
st.sidebar.markdown("---")
st.sidebar.subheader("Your History")
st.sidebar.text("Log in to see your history")


start = st.datetime_input("Enter the initial date:", min_value= datetime.datetime(2008,12,1,18,45), max_value="now")
end = st.datetime_input("Enter the later date:", min_value= datetime.datetime(2008,12,1,18,45), max_value="now")


values = ["ADBL","AHPC","AKJCL","AKPL","ALICL","API","BARUN","BFC","BPCL","CFCL","CGH","CHCL","CHDC","CHL",
          "CIT","CORBL","CZBIL","DHPL","EBL","EDBL","GBBL","GBIME","GFCL","GHL","GLH","GMFIL","GRDBL","GUFL","HBL","HDHPC",
          "HIDCL","HPPL","HURJA","ICFC","JBBL","JFL","JOSHI","KBL","KKHC","KPCL","KSBBL","LBBL","LEC","LICN","MBL",
          "MDB","MEN","MFIL","MHNL","MKJC","MLBL","MNBBL","MPFL","NABBC","NABIL","NBL","NFS","NGPL","NHDL","NHPC",
          "NICA","NIFRA","NLIC","NLICL","NMB","NRN","NYADI","OHL","PCBL","PFL","PMHPL","PPCL","PROFL","PRVU","RADHI",
          "RHPL","RLFL","RURU","SADBL","SAHAS","SANIMA","SAPDBL","SBI","SBL","SCB","SFCL","SHEL","SHINE",
          "SHL","SHPC","SIFC","SINDU","SJCL","SPC","SPDL","SSHL","TPC","TRH","UMHL","UMRH","UNHPL","UPCL",
          "UPPER"]
user_input = st.selectbox("Enter Stock Ticker eg: NABIL", values)
input = user_input + ".csv"
st.subheader(f"Data from {start} to {end}")


# Load the CSV file
base_path = (r'D:\Sumeedi\project_final\nepse-data\data\company-wise')
df = pd.read_csv(base_path +"\\"+ input)

# Save as a comma-delimited text file
df.to_csv(f'{user_input}.txt', sep=',', index=False)


df["published_date"] = pd.to_datetime(df["published_date"])
df1 = df[df["published_date"].between(start,end)]
df2 = df.dropna()

st.write(df1)


#describing data
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time chart")
#fig = go.Figure()
fig = plt.figure(figsize = (12,6))
plt.plot(df.published_date, df.close)
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r', label = '100 MA')
plt.plot(df.close, 'b', label = 'original closing price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r', label = '100 MA')
plt.plot(ma200,'g', label = '200 MA')
plt.plot(df.close, 'b', label = 'original closing price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)


# #splitting data into Training and Testing
# data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])    #[['open','high','low','close','traded_quantity']] for using all OHLCV to train
# data_testing = pd.DataFrame(df['close'][int(len(df)*0.70) : int(len(df))])    #or use .iloc as df[[]].iloc(int(len(df)*0.70): )

# scaler = MinMaxScaler(feature_range = (0,1))

# data_training_array = scaler.fit_transform(data_training)

# #load the training model (this model is non-linear regression model with RELU activation function)
# #model = load_model('LSTM_model')


# #testing part
# past_100_days = data_training.tail(100)
# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
# input_data = scaler.fit_transform(final_df)

# data = np.array(input_data).astype(float)


# x_test = []
# y_test = []

# for i in range(100,data.shape[0]):
#     x = data[i-100:i].reshape(100,1)
#     x_test.append(x)
#     y_test.append(data[i,0])     #i,0 in sense close column 1st maa xa vanera, here date is 5th col in csv file

# # st.write(data_training.shape)      2382,1  total
# # st.write(final_df.shape)           1121,1   training
# # st.write(input_data.shape)         1121,1    training
# #st.write(data_testing.shape)         1021,1    testing
# #st.write(x_test.shape)   list doesnot have attribute "shape"
# #st.write(input_data[:5])


# x_test = np.array(x_test)
# y_test = np.array(y_test)

# #y_predicted = model.predict(x_test)
# scaler = scaler.scale_

# scale_factor = 1/scaler[0]
# #y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor


# # #final graph
# # st.subheader('Predicted vs original')
# # fig2 = plt.figure(figsize=(12,6))
# # plt.plot(y_test,'b', label = 'original price')
# # plt.plot(y_predicted,'r', label = 'predicted price')
# # plt.xlabel('Time')
# # plt.ylabel('Price')
# # plt.legend()
# # st.pyplot(fig2)

