import streamlit as st
import mysql.connector
from datetime import datetime
import Home as h

#db configuration
DB_HOST = "localhost"
DB_NAME = "user_data"
DB_USER = "root"
DB_PASSWORD = ""


#db connection
def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )


#create history table
def create_history_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100),
            stock_name VARCHAR(100),
            prediction TEXT,
            requested_at DATETIME
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


create_history_table()

#main page(third)
st.set_page_config(
    page_title="Prediction History",
    page_icon="📜",
    layout="wide"
)

# custom css
st.markdown("""
<style>

.history-card {
    background-color: #1E1E1E;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border-left: 5px solid #00C897;
}

.metric-box {
    background-color: #262730;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

#authentication check
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "id" not in st.session_state:
    st.session_state.id = ""          
        


#block access if not logged in
if not st.session_state.logged_in:

    st.error("⚠️ Please login first to access prediction history.")

    st.stop()

# page header
st.title("📜 Prediction History Dashboard")

st.success(
    f"Logged in as: {st.session_state.username}"
)


# INSERT USERNAME, PREDICTION, STOCK NAME AND TIME INTO DATABASE
predicted_value = f"Rs.{st.session_state['prediction'][0]:.2f}"
conn = get_connection()
cur = conn.cursor()
if "prediction_saved" not in st.session_state:
    cur.execute("""
        INSERT INTO prediction_history
        (username, stock_name, prediction, requested_at)
        VALUES (%s, %s, %s, %s)
    """, (
        st.session_state.username,
        h.user_input,
        predicted_value,
        datetime.now()
    ))
    st.session_state.prediction_saved = True
    
conn.commit()
cur.close()
conn.close()

#fetch history
conn = get_connection()
cur = conn.cursor()

cur.execute("""
    SELECT stock_name, prediction, requested_at
    FROM prediction_history
    WHERE username = %s
    ORDER BY requested_at DESC
""", (st.session_state.username,))

history = cur.fetchall()

cur.close()
conn.close()

#dashbord
total_predictions = len(history)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{total_predictions}</h2>
        <p>Total Predictions Requested</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{datetime.now().strftime("%Y-%m-%d")}</h2>
        <p>Current Date</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

#display previous predictions(history)
if history:

    st.subheader("📊 Your Previous Prediction Requests")

    for item in history:

        stock_name = item[0]
        prediction = item[1]
        request_time = item[2]

        st.markdown(f"""
        <div class="history-card">

        <h4>📈 Stock: {stock_name}</h4>

        <p><b>Prediction:</b> {prediction}</p>

        <p><b>Requested At:</b> {request_time}</p>

        </div>
        """, unsafe_allow_html=True)

else:

    st.info("No prediction history found.")





if st.button("Clear History"):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM prediction_history
        WHERE username = %s
        """, (st.session_state.username,))
    conn.commit()
    cur.close()
    conn.close()

    st.success("Prediction history deleted")
    st.rerun()

    