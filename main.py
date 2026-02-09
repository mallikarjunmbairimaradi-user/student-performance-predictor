import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. SETUP & LOAD MODEL
# ---------------------------------------------------------
# FIXED: Added the closing bracket ')' at the end of this line
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Custom CSS to make it look nicer
st.markdown("""
    <style>
    /* Change the background color */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Style the big title */
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    /* Style the button */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the saved model
try:
    with open('student_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'student_model.pkl' not found. Please upload it to GitHub.")
    st.stop()

# ---------------------------------------------------------
# 2. THE FRONTEND INTERFACE
# ---------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.markdown("Enter student details below to predict the final **Exam Score**.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
    hours_studied = st.number_input("Hours Studied", min_value=0, max_value=100, value=20)
    previous_scores = st.number_input("Previous Score", min_value=0, max_value=100, value=75)
    tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1)

with col2:
    physical_activity = st.number_input("Physical Activity (Hrs/Week)", min_value=0, max_value=20, value=3)
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)

# ---------------------------------------------------------
# 3. BACKEND LOGIC (PREDICTION)
# ---------------------------------------------------------
if st.button("Predict Score"):
    # Create a DataFrame matching the model's training data
    input_data = pd.DataFrame(
        [[attendance, hours_studied, previous_scores, tutoring_sessions, 
          physical_activity, sleep_hours, access_to_resources, parental_involvement]],
        columns=['Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions', 
                 'Physical_Activity', 'Sleep_Hours', 'Access_to_Resources', 'Parental_Involvement']
    )

    # Make Prediction
    prediction = model.predict(input_data)[0]
    final_score = round(max(0, min(100, prediction)), 1)
    class_average = 67.2  # From your dataset

    # ---------------------------------------------------------
    # 4. DISPLAY RESULTS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Show the big number
    st.metric(label="Predicted Exam Score", value=f"{final_score}%", delta=f"{round(final_score - class_average, 1)} vs Avg")

    # Visualization: Bar Chart Comparison
    st.markdown("### 📊 Performance Comparison")
    
    # Create a simple bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#4CAF50' if final_score >= class_average else '#FF5252', '#2196F3']
    
    bars = ax.bar(['Student', 'Class Average'], [final_score, class_average], color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    
    # Add number labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%', ha='center', va='bottom')

    st.pyplot(fig)

    # Success/Warning Message
    if final_score >= 80:
        st.success("🌟 Excellent! This student is on track for a high distinction.")
    elif final_score >= 60:
        st.info("👍 Good job! The student is performing well.")
    else:
        st.warning("⚠️ Attention Needed: This student is at risk of underperforming.")
