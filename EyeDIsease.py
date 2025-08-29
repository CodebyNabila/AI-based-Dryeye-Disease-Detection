import streamlit as st
import pandas as pd
import datetime

# Sidebar navigation
st.sidebar.title("📚 Coursera-Lite Navigation")
page = st.sidebar.radio("Go to", ["Home", "Courses", "Progress", "Quizzes", "Profile", "Admin Dashboard", "About"])

# Dummy course data
courses = {
    "Python Basics": {"instructor": "Dr. Smith", "progress": 70},
    "Data Science 101": {"instructor": "Prof. Lee", "progress": 40},
    "UI/UX Design": {"instructor": "Jane Doe", "progress": 90},
}

# Home Page
if page == "Home":
    st.title("🎓 Coursera-Lite")
    st.subheader("Your Personalized Learning Platform")
    st.info("Welcome back! Continue where you left off 👇")

    for course, details in courses.items():
        st.write(f"**{course}** - Instructor: {details['instructor']}")
        st.progress(details['progress'] / 100)

# Courses Page
elif page == "Courses":
    st.title("📘 Available Courses")
    for course, details in courses.items():
        with st.expander(f"{course} (Instructor: {details['instructor']})"):
            st.write("📖 Course description goes here...")
            st.button(f"Enroll in {course}")

# Progress Page
elif page == "Progress":
    st.title("📊 Your Progress")
    df = pd.DataFrame(
        {"Course": list(courses.keys()), "Progress": [v["progress"] for v in courses.values()]}
    )
    st.bar_chart(df.set_index("Course"))

# Quizzes Page
elif page == "Quizzes":
    st.title("📝 Quick Quiz")
    q1 = st.radio("Python is ___ ?", ["Snake", "Programming Language", "Car"])
    if st.button("Submit Quiz"):
        if q1 == "Programming Language":
            st.success("✅ Correct!")
        else:
            st.error("❌ Wrong Answer!")

# Profile Page
elif page == "Profile":
    st.title("👤 Your Profile")
    name = st.text_input("Full Name", "Nabila Banu")
    email = st.text_input("Email", "example@email.com")
    dob = st.date_input("Date of Birth", datetime.date(2000, 1, 1))
    st.button("Update Profile")

# Admin Dashboard
elif page == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")
    st.write("Manage courses, users, and analytics here.")
    st.metric("Total Users", 128)
    st.metric("Total Courses", 10)
    st.metric("Avg Completion Rate", "65%")

# About Page
elif page == "About":
    st.title("ℹ️ About Coursera-Lite")
    st.markdown("""
    This is a **Streamlit-based LMS** inspired by Coursera.  
    Features include:  
    - 📚 Course browsing & enrollment  
    - 📊 Progress tracking  
    - 📝 Quizzes  
    - 👤 Profile management  
    - 🛠️ Admin Dashboard  
    """)
