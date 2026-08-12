
# ============================================================
# PHASE 8 - DEPLOYMENT
# PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# Streamlit Dashboard + Prediction API
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Learning System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "ml_model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"


@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


ml_model, encoder = load_models()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "feedback" not in st.session_state:
    st.session_state.feedback = []


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Personalized Learning Recommendation System")

st.markdown(
    """
    ### AI-Powered Personalized Learning Dashboard

    This system combines:

    - 📊 Student Performance Prediction
    - 📚 Content-Based Course Recommendation
    - 🧠 Adaptive Learning
    - 📅 Constraint-Based Scheduling
    - 🤖 AI Tutor
    - ⭐ Student Feedback
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Performance Prediction",
        "Course Recommendations",
        "Adaptive Learning",
        "Study Schedule",
        "AI Tutor",
        "Feedback",
        "API"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Student Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "System",
            "Active"
        )

    with col2:
        st.metric(
            "ML Model",
            "Loaded"
        )

    with col3:
        st.metric(
            "AI Tutor",
            "Available"
        )

    st.markdown("---")

    st.subheader("System Workflow")

    st.info(
        """
        Student Profile
        ↓
        Performance Prediction
        ↓
        Course Recommendation
        ↓
        Adaptive Learning
        ↓
        Study Scheduling
        ↓
        AI Tutor
        ↓
        Feedback
        """
    )


# ============================================================
# PERFORMANCE PREDICTION
# ============================================================

elif page == "Performance Prediction":

    st.header("📊 Student Performance Prediction")

    st.write(
        "Enter the student's academic and learning information."
    )

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.number_input(
            "Gender",
            min_value=0,
            max_value=1,
            value=0
        )

        region = st.number_input(
            "Region",
            min_value=0,
            value=0
        )

        highest_education = st.number_input(
            "Highest Education",
            min_value=0,
            value=0
        )

        imd_band = st.number_input(
            "IMD Band",
            min_value=0,
            value=0
        )

        age_band = st.number_input(
            "Age Band",
            min_value=0,
            value=0
        )

        disability = st.number_input(
            "Disability",
            min_value=0,
            max_value=1,
            value=0
        )

    with col2:

        num_of_prev_attempts = st.number_input(
            "Previous Attempts",
            min_value=0.0,
            value=0.0
        )

        studied_credits = st.number_input(
            "Studied Credits",
            min_value=0.0,
            value=60.0
        )

        total_assessments = st.number_input(
            "Total Assessments",
            min_value=0.0,
            value=5.0
        )

        average_score = st.number_input(
            "Average Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        highest_score = st.number_input(
            "Highest Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        lowest_score = st.number_input(
            "Lowest Score",
            min_value=0.0,
            max_value=100.0,
            value=30.0
        )

    with col3:

        total_clicks = st.number_input(
            "Total Clicks",
            min_value=0.0,
            value=100.0
        )

        average_clicks = st.number_input(
            "Average Clicks",
            min_value=0.0,
            value=10.0
        )

        max_clicks = st.number_input(
            "Maximum Clicks",
            min_value=0.0,
            value=20.0
        )

        resources_accessed = st.number_input(
            "Resources Accessed",
            min_value=0.0,
            value=5.0
        )

        active_days = st.number_input(
            "Active Days",
            min_value=0.0,
            value=10.0
        )

        registration_day = st.number_input(
            "Registration Day",
            min_value=0.0,
            value=0.0
        )

        course_duration = st.number_input(
            "Course Duration",
            min_value=0.0,
            value=240.0
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Student Performance",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [[
                gender,
                region,
                highest_education,
                imd_band,
                age_band,
                num_of_prev_attempts,
                studied_credits,
                disability,
                total_assessments,
                average_score,
                highest_score,
                lowest_score,
                total_clicks,
                average_clicks,
                max_clicks,
                resources_accessed,
                active_days,
                registration_day,
                course_duration
            ]],
            columns=[
                "gender",
                "region",
                "highest_education",
                "imd_band",
                "age_band",
                "num_of_prev_attempts",
                "studied_credits",
                "disability",
                "total_assessments",
                "average_score",
                "highest_score",
                "lowest_score",
                "total_clicks",
                "average_clicks",
                "max_clicks",
                "resources_accessed",
                "active_days",
                "registration_day",
                "course_duration"
            ]
        )

        try:

            prediction = ml_model.predict(input_data)

            result = encoder.inverse_transform(
                prediction.astype(int)
            )[0]

            st.session_state.prediction = result

            st.success(
                f"Predicted Student Result: **{result}**"
            )

            # ------------------------------------------------
            # PERFORMANCE INTERPRETATION
            # ------------------------------------------------

            if result == "Fail":

                st.warning(
                    "The student needs additional learning support."
                )

                st.markdown(
                    """
                    **Recommended actions:**
                    - Review weak topics
                    - Increase practice sessions
                    - Take frequent quizzes
                    - Use AI Tutor assistance
                    """
                )

            elif result == "Pass":

                st.success(
                    "The student is progressing satisfactorily."
                )

            elif result == "Distinction":

                st.success(
                    "Excellent performance! Advanced learning "
                    "resources are recommended."
                )

            elif result == "Withdrawn":

                st.warning(
                    "The student may be at risk of disengagement."
                )

        except Exception as e:

            st.error(
                f"Prediction error: {str(e)}"
            )


# ============================================================
# COURSE RECOMMENDATIONS
# ============================================================

elif page == "Course Recommendations":

    st.header("📚 Personalized Course Recommendations")

    st.info(
        "This section displays recommendations generated by Phase 3."
    )

    # --------------------------------------------------------
    # Load recommendations if available
    # --------------------------------------------------------

    recommendation_file = BASE_DIR / "recommendations.csv"

    if recommendation_file.exists():

        recommendations = pd.read_csv(
            recommendation_file
        )

        if "Similarity" in recommendations.columns:

            recommendations["Similarity"] = (
                recommendations["Similarity"]
                .round(3)
            )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            """
            recommendations.csv was not found.

            Export your Phase 3 recommendation DataFrame using:

            recommendations.to_csv(
                "recommendations.csv",
                index=False
            )
            """
        )


# ============================================================
# ADAPTIVE LEARNING
# ============================================================

elif page == "Adaptive Learning":

    st.header("🧠 Adaptive Learning Plan")

    adaptive_file = BASE_DIR / "adaptive_plan.csv"

    if adaptive_file.exists():

        adaptive_plan = pd.read_csv(
            adaptive_file
        )

        st.dataframe(
            adaptive_plan,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            """
            adaptive_plan.csv was not found.

            Export the Phase 4 result using:

            adaptive_plan.to_csv(
                "adaptive_plan.csv",
                index=False
            )
            """
        )


# ============================================================
# STUDY SCHEDULE
# ============================================================

elif page == "Study Schedule":

    st.header("📅 Personalized Study Schedule")

    st.write(
        """
        Select the days and time range when you are available
        for studying. The system will generate a personalized
        schedule based on your adaptive learning plan.
        """
    )

    # ========================================================
    # STEP 1: SELECT STUDY DAYS
    # ========================================================

    st.subheader("1️⃣ Select Available Study Days")

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    selected_days = st.multiselect(
        "Choose the days you are available:",
        options=days,
        default=["Monday", "Wednesday", "Friday"]
    )

    # ========================================================
    # STEP 2: SELECT START AND END TIME
    # ========================================================

    st.subheader("2️⃣ Select Available Study Time")

    col1, col2 = st.columns(2)

    with col1:

        study_start_time = st.time_input(
            "Study Start Time",
            value=pd.to_datetime("18:00").time()
        )

    with col2:

        study_end_time = st.time_input(
            "Study End Time",
            value=pd.to_datetime("21:00").time()
        )

    # ========================================================
    # STEP 3: DISPLAY SELECTED AVAILABILITY
    # ========================================================

    if selected_days:

        st.info(
            f"Available Days: {', '.join(selected_days)}"
        )

        st.info(
            f"Available Time: "
            f"{study_start_time.strftime('%I:%M %p')} - "
            f"{study_end_time.strftime('%I:%M %p')}"
        )

    # ========================================================
    # STEP 4: GENERATE SCHEDULE
    # ========================================================

    if st.button(
        "📅 Generate Personalized Schedule",
        type="primary"
    ):

        if len(selected_days) == 0:

            st.error(
                "Please select at least one study day."
            )

        elif study_start_time >= study_end_time:

            st.error(
                "Study end time must be later than study start time."
            )

        else:

            # ------------------------------------------------
            # Convert selected times to hours
            # ------------------------------------------------

            start_hour = (
                study_start_time.hour
                + study_start_time.minute / 60
            )

            end_hour = (
                study_end_time.hour
                + study_end_time.minute / 60
            )

            # ------------------------------------------------
            # Convert to 15-minute slots
            # ------------------------------------------------

            start_slot = int(start_hour * 4)
            end_slot = int(end_hour * 4)

            # ------------------------------------------------
            # Generate schedule from adaptive plan
            # ------------------------------------------------

            try:

                schedule_rows = []

                # ------------------------------------------------
                # Calculate total available minutes per day
                # ------------------------------------------------

                available_minutes_per_day = (
                    end_hour - start_hour
                ) * 60

                # ------------------------------------------------
                # Current day pointer
                # ------------------------------------------------

                day_index = 0

                current_day = selected_days[day_index]

                current_slot = start_slot

                # ------------------------------------------------
                # Schedule each adaptive learning resource
                # ------------------------------------------------

                for _, row in adaptive_plan.iterrows():

                    # --------------------------------------------
                    # Estimated learning time
                    # --------------------------------------------

                    estimated_time = row.get(
                        "Estimated Time",
                        30
                    )

                    try:
                        estimated_time = float(
                            estimated_time
                        )
                    except:
                        estimated_time = 30

                    # --------------------------------------------
                    # Convert minutes to 15-minute slots
                    # --------------------------------------------

                    required_slots = max(
                        1,
                        int(
                            np.ceil(
                                estimated_time / 15
                            )
                        )
                    )

                    duration_minutes = (
                        required_slots * 15
                    )

                    # --------------------------------------------
                    # Check whether resource fits current day
                    # --------------------------------------------

                    if (
                        current_slot + required_slots
                        > end_slot
                    ):

                        # Move to next selected day
                        day_index += 1

                        if day_index >= len(selected_days):

                            day_index = 0

                        current_day = selected_days[
                            day_index
                        ]

                        current_slot = start_slot

                    # --------------------------------------------
                    # Calculate start/end time
                    # --------------------------------------------

                    start_minutes = (
                        current_slot * 15
                    )

                    end_minutes = (
                        (current_slot + required_slots)
                        * 15
                    )

                    start_h = start_minutes // 60
                    start_m = start_minutes % 60

                    end_h = end_minutes // 60
                    end_m = end_minutes % 60

                    start_time = (
                        f"{int(start_h):02d}:"
                        f"{int(start_m):02d}"
                    )

                    end_time = (
                        f"{int(end_h):02d}:"
                        f"{int(end_m):02d}"
                    )

                    # --------------------------------------------
                    # Add schedule row
                    # --------------------------------------------

                    schedule_rows.append(
                        {
                            "Day": current_day,
                            "Start Time": start_time,
                            "End Time": end_time,
                            "Course": row.get(
                                "course",
                                ""
                            ),
                            "Learning Resource": row.get(
                                "title",
                                ""
                            ),
                            "Difficulty": row.get(
                                "difficulty",
                                ""
                            ),
                            "Duration (min)": duration_minutes,
                            "Learning Level": row.get(
                                "Learning Level",
                                ""
                            )
                        }
                    )

                    # --------------------------------------------
                    # Move to next available slot
                    # --------------------------------------------

                    current_slot += required_slots

                # ------------------------------------------------
                # Convert to DataFrame
                # ------------------------------------------------

                generated_schedule = pd.DataFrame(
                    schedule_rows
                )

                # ------------------------------------------------
                # Save in session state
                # ------------------------------------------------

                st.session_state.generated_schedule = (
                    generated_schedule
                )

                # ------------------------------------------------
                # Display result
                # ------------------------------------------------

                st.success(
                    "✓ Personalized study schedule generated!"
                )

                st.subheader(
                    "📋 Your Personalized Schedule"
                )

                st.dataframe(
                    generated_schedule,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:

                st.error(
                    f"Schedule generation error: {str(e)}"
                )

    # ========================================================
    # STEP 5: SHOW PREVIOUSLY GENERATED SCHEDULE
    # ========================================================

    if (
        "generated_schedule"
        in st.session_state
    ):

        st.subheader(
            "📋 Current Study Schedule"
        )

        st.dataframe(
            st.session_state.generated_schedule,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# AI TUTOR
# ============================================================

elif page == "AI Tutor":

    st.header("🤖 AI Tutor")

    st.write(
        "Ask questions about your learning topics."
    )

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Explain CNN in simple terms."
    )

    if st.button("Ask AI Tutor"):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # Simple local tutor response
            # ------------------------------------------------

            question_lower = question.lower()

            if "cnn" in question_lower:

                response = """
                CNN stands for Convolutional Neural Network.

                It is a deep learning model mainly used for
                image processing and computer vision.

                Main components:
                1. Convolution layer
                2. Pooling layer
                3. Fully connected layer
                4. Output layer
                """

            elif "lstm" in question_lower:

                response = """
                LSTM stands for Long Short-Term Memory.

                It is a type of recurrent neural network designed
                to learn long-term dependencies in sequential data.

                LSTM is commonly used for:
                - Time-series prediction
                - Text generation
                - Speech processing
                - Sentiment analysis
                """

            elif "machine learning" in question_lower:

                response = """
                Machine Learning is a branch of Artificial
                Intelligence where systems learn patterns from
                data and use those patterns to make predictions
                or decisions.
                """

            else:

                response = """
                The AI Tutor recommends reviewing the relevant
                learning material and practicing the topic.

                You can also ask about specific concepts such as
                CNN, LSTM, Machine Learning, Deep Learning,
                classification, or neural networks.
                """

            st.success("AI Tutor Response")

            st.write(response)


# ============================================================
# FEEDBACK
# ============================================================

elif page == "Feedback":

    st.header("⭐ Student Feedback")

    rating = st.slider(
        "Rate the Personalized Learning System",
        min_value=1,
        max_value=5,
        value=5
    )

    feedback = st.text_area(
        "Enter your feedback"
    )

    if st.button("Submit Feedback"):

        feedback_record = {
            "Rating": rating,
            "Feedback": feedback,
            "Predicted Result":
                st.session_state.prediction
        }

        st.session_state.feedback.append(
            feedback_record
        )

        st.success(
            "✓ Thank you! Your feedback has been recorded."
        )

    if len(st.session_state.feedback) > 0:

        st.subheader("Feedback Records")

        feedback_df = pd.DataFrame(
            st.session_state.feedback
        )

        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# API INFORMATION
# ============================================================

elif page == "API":

    st.header("🔌 Prediction API")

    st.info(
        """
        The trained Phase 2 model can also be exposed as an API.
        """
    )

    st.subheader("API Input Format")

    example_input = {
        "gender": 0,
        "region": 0,
        "highest_education": 2,
        "imd_band": 5,
        "age_band": 2,
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": 0,
        "total_assessments": 5,
        "average_score": 65,
        "highest_score": 85,
        "lowest_score": 40,
        "total_clicks": 100,
        "average_clicks": 10,
        "max_clicks": 20,
        "resources_accessed": 8,
        "active_days": 15,
        "registration_day": 0,
        "course_duration": 240
    }

    st.json(example_input)

    st.subheader("Expected Response")

    st.json(
        {
            "prediction": "Pass"
        }
    )

    st.warning(
        """
        Streamlit is primarily a dashboard framework.
        For a production REST API, use FastAPI as a separate
        service alongside this Streamlit application.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Personalized Learning Recommendation System | "
    "M.Tech Project"
)

