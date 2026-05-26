import streamlit as st
import numpy as np
import regressionModel as rm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Home as h
# from keras.models import load_model
import joblib
import re
import os
import mysql.connector
import bcrypt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Page configuration
st.set_page_config(
    page_title="Login to Predict",
    page_icon="🔐",
    layout="centered"
)

# DB connection

DB_HOST = "localhost"
DB_NAME = "user_data"
DB_USER = "root"
DB_PASSWORD = ""


def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

# create table
def create_users_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        email VARCHAR(200) UNIQUE NOT NULL,
        password TEXT NOT NULL
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


create_users_table()

# password hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


# register user
def register_user(username, email, password):
    try:
        conn = get_connection()
        cur = conn.cursor()

        hashed_pw = hash_password(password)
        
        cur.execute("""
            INSERT INTO users (username, email, password)
            VALUES (%s, %s, %s)
        """, (username, email, hashed_pw))

        conn.commit()

        cur.close()
        conn.close()

        return True

    except Exception as e:
        st.error(f"Registration Error: {e}")
        return False


# login user
def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT password FROM users
        WHERE username = %s
    """, (username,))

    result = cur.fetchone()

    cur.close()
    conn.close()

    if result:
        stored_password = result[0]

        if verify_password(password, stored_password):
            return True

    return False


# session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "id" not in st.session_state:
    st.session_state.id = ""

# custom css
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.auth-box {
    padding: 2rem;
    border-radius: 15px;
    background-color: #1E1E1E;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}

.feature-card {
    padding: 1rem;
    border-radius: 12px;
    background-color: #262730;
    margin-bottom: 1rem;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# main page(second one)
st.title("You can now predict")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

#register
if choice == "Register":

    st.markdown('<div class="auth-box">', unsafe_allow_html=True)

    st.subheader("Create New Account")

    new_user = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")


    if st.button("Create Account"):

        if new_user and new_email and new_password:
            if not re.match(r'^[a-zA-Z0-9]+$', new_user):
                st.info("Username can only contain letters and numbers")
            else:
                success = register_user(
                    new_user,
                    new_email,
                    new_password
                )

                if success:
                    st.success("Account Created Successfully!")
                    st.info("Go to Login Menu to login.")

        else:
            st.warning("Please fill all fields.")

    st.markdown('</div>', unsafe_allow_html=True)

#login
elif choice == "Login":

    if not st.session_state.logged_in:

        st.markdown('<div class="auth-box">', unsafe_allow_html=True)

        st.subheader("Login to Continue")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if login_user(username, password):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.success("Login Successful!")
                st.rerun()

            else:
                st.error("Invalid Username or Password")

        st.markdown('</div>', unsafe_allow_html=True)

    else:

        st.success(
            f"Welcome {st.session_state.username} "
        )

        st.subheader("💎 Premium Features")

        st.markdown("""
        <div class="feature-card">
            <h4>📈 Advanced Analytics</h4>
            <p>Premium market insights and data.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>AI Predictions</h4>
            <p>AI-powered forecasting tools.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>🔒 Exclusive Premium Content</h4>
            <p>Only accessible to logged-in users.</p>
        </div>
        """, unsafe_allow_html=True)

        #buttons for prediction
        st.sidebar.header("Predict the next day's closing")

        # Prediction Logic for linear regression model
        st.subheader("Next Day Prediction")
        if st.button("Predict Closing Price(using Linear Regression)"):
            # Training on the 100 rows in csv
            df_last_100 = h.df.tail(100)
            X = df_last_100[['open', 'high', 'low', 'traded_quantity']].values
            y = df_last_100['close'].values
                
            # Normalize
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_norm = (X - X_mean) / X_std
                
            # Train model
            model = rm.CustomLinearRegression(lr=0.1, iterations=2000)
            model.fit(X_norm, y)
            last_day = X_norm[-1].reshape(1, -1)
            prediction = model.predict(last_day)
            st.session_state["prediction"] = prediction
            # print(last_day)

            # actual_close = h.df.tail(1).close
            # # print(h.df.tail(1).close)
            # mse = mean_squared_error(actual_close, prediction)
            # rmse = np.sqrt(mse)
            # mae = mean_absolute_error(actual_close, prediction)
            # r2 = r2_score(actual_close, prediction)
            # print("MSE:", mse)
            # print("RMSE:", rmse)
            # print("MAE:", mae)
            # print("R2:", r2)
            # #final graph
            # st.subheader('Predicted vs original for last 100 days')
            # ax = plt.subplot()
            # fig3 = plt.figure(figsize=(12,6))
            # plt.plot(prediction,'b', label = 'predicted price')
            # plt.plot(y, 'r', label = 'original price')
            # plt.xlabel('Time')
            # plt.ylabel('Price')
            # plt.legend()
            # st.pyplot(fig3)
            st.metric("Predicted Close", f"Rs.{prediction[0]:.2f}")

        if st.button("Predict Closing Price(using LSTM Model)"):
            #splitting data into Training and Testing
            data_training = pd.DataFrame(h.df['close'][0:int(len(h.df)*0.70)])    #[['open','high','low','close','traded_quantity']] for using all OHLCV to train
            data_testing = pd.DataFrame(h.df['close'][int(len(h.df)*0.70) : int(len(h.df))])    #or use .iloc as df[[]].iloc(int(len(df)*0.70): )

            scaler = MinMaxScaler(feature_range = (0,1))

            data_training_array = scaler.fit_transform(data_training)

            #load the training model (this model is non-linear regression model with RELU activation function)
            # model = load_model('LSTM_model')
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
                y_test.append(data[i,0]) 
            
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            y_predicted = model.predict(h.x_test)
            scaler = scaler.scale_
            scale_factor = 1/scaler[0]
            y_predicted = y_predicted * h.scale_factor
            y_test = y_test * scale_factor


            #final graph
            st.subheader('Predicted vs original')
            fig2 = plt.figure(figsize=(12,6))
            plt.plot(h.y_test,'b', label = 'original price')
            plt.plot(y_predicted,'r', label = 'predicted price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)

        if st.button("Predict Closing Price(using SVM Model)"):
            model=joblib.load("nepse-data/src/SVM_model.pkl")

            latest_100_days = h.scaled_data[-100:]

            X = np.array([latest_100_days])

            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

            prediction = model.predict(X)

            dummy = np.zeros((1,5))

            dummy[0,3] = prediction[0]

            prediction = h.scaler.inverse_transform(dummy)

            pred = st.session_state['prediction'].item()

            prediction = f"Rs.{pred:.2f}"

            st.session_state["prediction"] = prediction[0,3]


            st.metric("Predicted next close:", prediction)

        if st.button("Predict Closing Price(using Random Forest Model)"):
            model=joblib.load("nepse-data/src/RandomForest.pkl")

            latest_100_days = h.scaled_data[-100:]

            X = np.array([latest_100_days])

            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

            prediction = model.predict(X)

            dummy = np.zeros((1,5))

            dummy[0,3] = prediction[0]

            actual_price = h.scaler.inverse_transform(dummy)
            closing = actual_price.flatten()    # yo garna ni mildaina rey!!!!!!!!!!


            st.metric("Predicted next close:", closing(2))
            # st.write(type(actual_price))

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
         