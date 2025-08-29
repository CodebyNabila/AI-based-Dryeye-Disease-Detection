import streamlit as st
import sqlite3
import bcrypt
import jwt
import datetime
import google.generativeai as genai
import time

# Set up Gemini AI (Replace with your API key)
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Database Setup
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    email TEXT UNIQUE
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    symptoms TEXT,
    prediction TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Secret Key for JWT Token
SECRET_KEY = "your_secret_key"

# Function: Hash Password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function: Verify Password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# Function: Generate JWT Token
def generate_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# Function: Decode JWT Token
def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["username"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Function: Register User
def register_user(username, password, email):
    try:
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, hash_password(password), email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Function: Authenticate User
def authenticate_user(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user and verify_password(password, user[0]):
        return generate_token(username)
    return None

# Function: Predict Dry Eye Disease (Using Gemini AI)
def predict_dry_eye_disease(symptoms):
    prompt = f"Analyze these symptoms: {symptoms}. Predict Dry Eye Disease severity and recommend treatment."
    response = genai.generate_text(prompt)
    return response.result if response else "Error generating response."

# Function: Save Prediction to Database
def save_prediction(username, symptoms, prediction):
    c.execute("INSERT INTO predictions (username, symptoms, prediction) VALUES (?, ?, ?)", (username, symptoms, prediction))
    conn.commit()

# Function: Get Prediction History
def get_prediction_history(username):
    c.execute("SELECT symptoms, prediction, timestamp FROM predictions WHERE username = ? ORDER BY timestamp DESC", (username,))
    return c.fetchall()

# UI Styling
st.set_page_config(page_title="Dry Eye Disease AI", page_icon="👁️", layout="centered")
st.markdown("<h2 style='text-align: center; color: blue;'>👁️ AI-Based Dry Eye Disease Detection</h2>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3d/Medical_cross_icon.png", width=100)
menu = st.sidebar.radio("Navigation", ["Login", "Register", "Prediction", "Profile", "History"])

# Register Page
if menu == "Register":
    st.subheader("Register")
    new_username = st.text_input("Create Username", placeholder="Choose a username")
    new_email = st.text_input("Email", placeholder="Enter your email")
    new_password = st.text_input("Create Password", type="password", placeholder="Choose a password")

    if st.button("Register"):
        if register_user(new_username, new_password, new_email):
            st.success("Registration successful! You can now log in.")
        else:
            st.error("Username or email already taken!")
            
# Login Page
elif menu == "Login":
    st.subheader("Login")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login"):
        token = authenticate_user(username, password)
        if token:
            st.success(f"Welcome, {username}!")
            st.session_state["token"] = token
        else:
            st.error("Invalid username or password!")

# Prediction Page (Requires Login)
elif menu == "Prediction":
    if "token" in st.session_state and decode_token(st.session_state["token"]):
        st.subheader("AI Dry Eye Disease Prediction")
        symptoms = st.text_area("Enter Symptoms (e.g., redness, irritation, blurry vision)")

        if st.button("Predict"):
            if symptoms:
                with st.spinner("Processing..."):
                    time.sleep(2)
                    prediction_result = predict_dry_eye_disease(symptoms)
                    save_prediction(decode_token(st.session_state["token"]), symptoms, prediction_result)
                st.subheader("AI Prediction:")
                st.write(prediction_result)
            else:
                st.error("Please enter symptoms.")

        # Logout Button
        if st.button("Logout"):
            del st.session_state["token"]
            st.success("Logged out successfully!")

    else:
        st.warning("Please log in to access this feature.")

# Profile Page (Allows Users to Update Their Info)
elif menu == "Profile":
    if "token" in st.session_state and decode_token(st.session_state["token"]):
        st.subheader("👤 Profile")
        username = decode_token(st.session_state["token"])
        st.write(f"**Username:** {username}")
        
        new_password = st.text_input("Change Password", type="password", placeholder="Enter new password")
        if st.button("Update Password") and new_password:
            hashed_password = hash_password(new_password)
            c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_password, username))
            conn.commit()
            st.success("Password updated successfully!")

        # Logout Button
        if st.button("Logout"):
            del st.session_state["token"]
            st.success("Logged out successfully!")

    else:
        st.warning("Please log in to view your profile.")

# History Page (Shows Past Predictions)
elif menu == "History":
    if "token" in st.session_state and decode_token(st.session_state["token"]):
        st.subheader("Prediction History")
        history = get_prediction_history(decode_token(st.session_state["token"]))
        
        if history:
            for symptoms, prediction, timestamp in history:
                with st.expander(f" {timestamp}"):
                    st.write(f"**Symptoms:** {symptoms}")
                    st.write(f"**Prediction:** {prediction}")
        else:
            st.info("No predictions found.")
        
        # Logout Button
        if st.button("Logout"):
            del st.session_state["token"]
            st.success("Logged out successfully!")

    else:
        st.warning("Please log in to view history.")