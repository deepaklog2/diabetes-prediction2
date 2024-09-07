import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps, ImageDraw
import logging
import PyPDF2
import re
import os
from dotenv import load_dotenv
import hashlib
from io import BytesIO
import random

# Load environment variables from .env file
load_dotenv()

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Fetch environment variables
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
SENDER_EMAIL = "16gomathimsc@gmail.com"
USER_DB_FILE = 'users.csv'

# Initialize user database
def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        df = pd.DataFrame(columns=['username', 'password', 'email', 'image', 'user_id'])
        df.to_csv(USER_DB_FILE, index=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, email, image):
    df = pd.read_csv(USER_DB_FILE) if os.path.exists(USER_DB_FILE) and os.path.getsize(USER_DB_FILE) > 0 else pd.DataFrame(columns=['username', 'password', 'email', 'image', 'user_id'])
    if username not in df['username'].values:
        hashed_password = hash_password(password)
        user_id = f"{random.randint(10000, 99999)}"  # Generate a 5-digit ID
        new_user = pd.DataFrame([[username, hashed_password, email, image, user_id]], columns=['username', 'password', 'email', 'image', 'user_id'])
        df = pd.concat([df, new_user], ignore_index=True)
        df.to_csv(USER_DB_FILE, index=False)
        return True
    return False

def authenticate_user(username, password):
    if os.path.exists(USER_DB_FILE) and os.path.getsize(USER_DB_FILE) > 0:
        df = pd.read_csv(USER_DB_FILE)
    else:
        return False
    if username in df['username'].values:
        hashed_password = hash_password(password)
        return hashed_password in df[df['username'] == username]['password'].values
    return False

def login_page():
    st.title("Login")

    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(login_username, login_password):
            st.session_state.logged_in = True
            st.session_state.username = login_username
            st.session_state.page = "app"  # Redirect to app page
        else:
            st.error("Invalid username or password.")

def register_page():
    st.title("Register")

    reg_username = st.text_input("New Username")
    reg_password = st.text_input("New Password", type="password")
    reg_email = st.text_input("Email")
    uploaded_image = st.file_uploader("Upload Profile Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
    else:
        image_bytes = b''

    if st.button("Register"):
        if add_user(reg_username, reg_password, reg_email, image_bytes):
            st.success("User registered successfully! Please log in.")
            st.session_state.page = "login"  # Go to login page after registration
        else:
            st.error("Username already exists.")
    
    if st.button("Go to Login"):
        st.session_state.page = "login"  # Allow users to go to login page

def check_login():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.write("Please log in to access the app.")
        st.write("[Log In](http://localhost:8501)")  # Correct port number
        st.stop()

def load_data():
    try:
        logger.debug("Loading dataset...")
        diabetes_df = pd.read_csv('diabetes.csv')
        logger.debug("Dataset loaded successfully.")
        return diabetes_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        logger.error(f"FileNotFoundError: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
        st.stop()

def preprocess_data(df):
    try:
        logger.debug("Preprocessing data...")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.debug("Data preprocessing completed.")
        return X_scaled, y, scaler
    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        logger.error(f"Preprocessing error: {e}")
        st.stop()

def train_model(X_train, y_train):
    try:
        logger.debug("Training model...")
        model = svm.SVC(kernel='linear')
        model.fit(X_train, y_train)
        logger.debug("Model training completed.")
        return model
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        logger.error(f"Model training error: {e}")
        st.stop()

def get_precautionary_advice(features):
    glucose, bp, insulin, skinthickness, bmi, dpf, age = features[1:8]
    
    advice = []

    if glucose > 125:
        advice.append("• High glucose levels can be managed by reducing sugar intake, eating a balanced diet, and increasing physical activity.")
        advice.append("• Regular monitoring of blood sugar levels is important.")
        advice.append("• Consult a healthcare professional for personalized advice.")
        advice.append("• Consider joining a diabetes education program for more guidance.")
        advice.append("• Monitor glucose levels regularly to prevent complications.")

    if insulin > 25:
        advice.append("• High insulin levels can be controlled by following a healthy diet, maintaining a healthy weight, and avoiding excessive sugar intake.")
        advice.append("• Consider regular physical activity to improve insulin sensitivity.")
        advice.append("• Consult a dietitian for a personalized meal plan.")
        advice.append("• Discuss with a healthcare provider if medication adjustments are needed.")

    if dpf > 0.5:
        advice.append("• A high Diabetes Pedigree Function indicates a family history of diabetes. Maintain a healthy lifestyle and get regular check-ups.")
        advice.append("• Consider genetic counseling if there is a significant family history.")
        advice.append("• Stay informed about diabetes prevention strategies.")
        advice.append("• Monitor your health regularly for early signs of diabetes.")

    if age > 60:
        advice.append("• Older age can increase the risk of diabetes. Regular health check-ups and maintaining a healthy lifestyle are important.")
        advice.append("• Ensure regular monitoring of blood glucose levels and consult a healthcare provider for appropriate measures.")

    if glucose <= 125 and insulin <= 25 and dpf <= 0.5 and age <= 60:
        if bp > 80:
            advice.append("• High blood pressure can be managed by reducing salt intake, exercising regularly, and avoiding stress.")
            advice.append("• Monitor blood pressure frequently and take medications if prescribed.")
            advice.append("• Maintain a healthy weight and reduce alcohol consumption.")
            advice.append("• Regular check-ups with a healthcare provider are recommended.")
        
        if skinthickness > 30:
            advice.append("• Increased skin thickness can be managed by improving diet and increasing physical activity.")
            advice.append("• Monitor skin changes and consult a dermatologist if needed.")
            advice.append("• Regular exercise and a balanced diet are key.")
            advice.append("• Check for other potential underlying conditions with a healthcare provider.")
        
        if bmi > 30:
            advice.append("• A high BMI indicates obesity. Consider a balanced diet and regular exercise to maintain a healthy weight.")
            advice.append("• Aim for gradual weight loss through lifestyle changes.")
            advice.append("• Consult a healthcare provider for a weight management plan.")
            advice.append("• Avoid fad diets and focus on sustainable changes.")
            advice.append("• Incorporate both aerobic and strength training exercises.")
    
    return advice

def process_uploaded_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfFileReader(uploaded_file)
        first_page = reader.getPage(0).extract_text()
        
        values = {
            'Pregnancies': None,
            'Glucose': None,
            'Blood Pressure': None,
            'Skin Thickness': None,
            'Insulin': None,
            'BMI': None,
            'Diabetes Pedigree Function': None,
            'Age': None
        }
        
        for key in values.keys():
            match = re.search(f'{key}:\s*(\d+\.?\d*)', first_page)
            if match:
                values[key] = float(match.group(1))
        
        return list(values.values())
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        st.error("Error processing PDF. Please ensure the format is correct.")
        return None

def send_email(subject, body, to_email):
    try:
        email_content = f"Subject: {subject}\n\n{body}"
        logger.info(f"Email to be sent to {to_email}:\n{email_content}")
        st.success('Your message has been prepared! Please check the console for the email content.')
    except Exception as e:
        logger.error(f"Error preparing email: {e}")
        st.error("Failed to prepare email. Please try again.")

def main():
    init_user_db()

    if 'page' not in st.session_state:
        st.session_state.page = "register"  # Start with registration page

    if 'logged_in' in st.session_state and st.session_state.logged_in:
        st.session_state.page = "app"  # Redirect to app.py when logged in

    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "register":
        register_page()
    elif st.session_state.page == "app":
        app()

def app():
    check_login()

    diabetes_df = load_data()
    X_scaled, y, scaler = preprocess_data(diabetes_df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
    model = train_model(X_train, y_train)

    # Display profile image and details in the sidebar
    df = pd.read_csv(USER_DB_FILE)
    user_profile = df[df['username'] == st.session_state.username].iloc[0]

    # User ID and profile details
    user_id = user_profile['user_id']
    profile_image = user_profile['image']

    # Sidebar for user profile
    with st.sidebar.expander("User Profile", expanded=True):
        st.write(f"**Username:** {st.session_state.username}")
        st.write(f"**Email:** {user_profile['email']}")
        st.write(f"**User ID:** {user_id}")

        if profile_image and isinstance(profile_image, (bytes, bytearray)):
            image_bytes = BytesIO(profile_image)
            try:
                img = Image.open(image_bytes)
                img = img.resize((100, 100))  # Adjust size as needed
                img = ImageOps.fit(img, (100, 100), method=Image.LANCZOS)
                # Create circular mask
                mask = Image.new('L', (100, 100), 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, 100, 100), fill=255)
                img.putalpha(mask)
                st.image(img, caption='Profile Picture', use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying profile image: {e}")
                logger.error(f"Profile image display error: {e}")

        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.experimental_rerun()  # Refresh the page to redirect to login

    st.title('Diabetes Prediction')

    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    logger.debug(f"Input data: {input_data}")

    reshaped_input_data = np.array(input_data).reshape(1, -1)
    scaled_input_data = scaler.transform(reshaped_input_data)
    prediction = model.predict(scaled_input_data)

    if prediction[0] == 1:
        st.write("**Prediction:** You are at risk of diabetes.")
        advice = get_precautionary_advice(input_data)
        st.write("**Precautionary Advice:**")
        for line in advice:
            st.write(line)
    else:
        st.write("**Prediction:** You are not at risk of diabetes. Great job maintaining your health!")

    # Move PDF upload and contact form to main area
    st.write("**Upload a PDF**")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_data = process_uploaded_pdf(uploaded_file)
        if pdf_data:
            st.write("Extracted PDF data:", pdf_data)
            scaled_pdf_data = scaler.transform([pdf_data])
            pdf_prediction = model.predict(scaled_pdf_data)
            if pdf_prediction[0] == 1:
                st.write("**PDF Prediction:** Risk of diabetes detected.")
                advice = get_precautionary_advice(pdf_data)
                st.write("**Precautionary Advice from PDF:**")
                for line in advice:
                    st.write(line)
            else:
                st.write("**PDF Prediction:** No risk of diabetes detected.")

    st.write("**Contact Us**")
    subject = st.text_input('Subject')
    message = st.text_area('Message')
    email = st.text_input('Your Email')

    if st.button('Send'):
        if subject and message and email:
            send_email(subject, message, SENDER_EMAIL)
        else:
            st.error("Please fill out all fields.")

if __name__ == "__main__":
    main()