
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
    # USER-FRIENDLY CATEGORY MAPPINGS
    # --------------------------------------------------------

    # Gender encoding used by OULAD / LabelEncoder
    gender_map = {
        "♀ Female": 0,
        "♂ Male": 1
    }

    # Region encoding
    region_map = {
        "East Anglian Region": 0,
        "East Midlands Region": 1,
        "Ireland": 2,
        "London Region": 3,
        "North Region": 4,
        "North Western Region": 5,
        "Scotland": 6,
        "South East Region": 7,
        "South Region": 8,
        "South West Region": 9,
        "Wales": 10,
        "West Midlands Region": 11,
        "Yorkshire Region": 12
    }

    # Highest education encoding
    education_map = {
        "A Level or Equivalent": 0,
        "HE Qualification": 1,
        "Lower Than A Level": 2,
        "No Formal Qualification": 3,
        "Post Graduate Qualification": 4
    }

    # IMD band encoding
    imd_map = {
        "0-10%": 0,
        "10-20%": 1,
        "20-30%": 2,
        "30-40%": 3,
        "40-50%": 4,
        "50-60%": 5,
        "60-70%": 6,
        "70-80%": 7,
        "80-90%": 8,
        "90-100%": 9,
        "Missing / Unknown": 10
    }

    # Disability encoding
    disability_map = {
        "No": 0,
        "Yes": 1
    }

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:

        # ----------------------------------------------------
        # GENDER
        # ----------------------------------------------------

        gender_label = st.radio(
            "Gender",
            options=[
                "♀ Female",
                "♂ Male"
            ],
            horizontal=True
        )

        gender = gender_map[gender_label]

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        age = st.selectbox(
            "Age",
            options=list(range(18, 71)),
            index=2
        )

        # Convert actual age to OULAD age band
        if age < 35:
            age_band = 0
        elif age < 55:
            age_band = 1
        else:
            age_band = 2

        # ----------------------------------------------------
        # REGION
        # ----------------------------------------------------

        region_label = st.selectbox(
            "Region",
            options=list(region_map.keys())
        )

        region = region_map[region_label]

        # ----------------------------------------------------
        # HIGHEST EDUCATION
        # ----------------------------------------------------

        education_label = st.selectbox(
            "Highest Education",
            options=list(education_map.keys())
        )

        highest_education = education_map[
            education_label
        ]

        # ----------------------------------------------------
        # IMD BAND
        # ----------------------------------------------------

        imd_label = st.selectbox(
            "IMD Band",
            options=list(imd_map.keys())
        )

        imd_band = imd_map[imd_label]

        # ----------------------------------------------------
        # DISABILITY
        # ----------------------------------------------------

        disability_label = st.radio(
            "Disability",
            options=[
                "No",
                "Yes"
            ],
            horizontal=True
        )

        disability = disability_map[
            disability_label
        ]

    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        num_of_prev_attempts = st.number_input(
            "Previous Attempts",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        studied_credits = st.number_input(
            "Studied Credits",
            min_value=0.0,
            value=60.0,
            step=10.0
        )

        total_assessments = st.number_input(
            "Total Assessments",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        average_score = st.number_input(
            "Average Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        highest_score = st.number_input(
            "Highest Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        lowest_score = st.number_input(
            "Lowest Score",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )

    # ========================================================
    # COLUMN 3
    # ========================================================

    with col3:

        total_clicks = st.number_input(
            "Total Clicks",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        average_clicks = st.number_input(
            "Average Clicks",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        max_clicks = st.number_input(
            "Maximum Clicks",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

        resources_accessed = st.number_input(
            "Resources Accessed",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        active_days = st.number_input(
            "Active Days",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        registration_day = st.number_input(
            "Registration Day",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        course_duration = st.number_input(
            "Course Duration",
            min_value=0.0,
            value=240.0,
            step=1.0
        )

    # --------------------------------------------------------
    # SHOW INTERNAL AGE BAND INFORMATION
    # --------------------------------------------------------

    with st.expander("ℹ️ Age information"):

        if age < 35:
            age_band_text = "0–35"
        elif age < 55:
            age_band_text = "35–55"
        else:
            age_band_text = "55+"

        st.write(
            f"Selected Age: **{age} years**"
        )

        st.write(
            f"Model Age Band: **{age_band_text}**"
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

            prediction = ml_model.predict(
                input_data
            )

            result = encoder.inverse_transform(
                prediction.astype(int)
            )[0]

            st.session_state.prediction = result

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

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

                    - 📚 Review weak topics
                    - 📝 Increase practice sessions
                    - ❓ Take frequent quizzes
                    - 🤖 Use AI Tutor assistance
                    - 📅 Follow the personalized study schedule
                    """
                )

            elif result == "Pass":

                st.success(
                    "The student is progressing satisfactorily."
                )

                st.markdown(
                    """
                    **Recommended actions:**

                    - Continue the current learning plan
                    - Practice regularly
                    - Follow the recommended study schedule
                    - Use quizzes to maintain performance
                    """
                )

            elif result == "Distinction":

                st.success(
                    "Excellent performance! Advanced learning "
                    "resources are recommended."
                )

                st.markdown(
                    """
                    **Recommended actions:**

                    - 🚀 Explore advanced topics
                    - 📚 Attempt challenging resources
                    - 🧠 Practice advanced problems
                    - 🎯 Maintain consistent learning
                    """
                )

            elif result == "Withdrawn":

                st.warning(
                    "The student may be at risk of disengagement."
                )

                st.markdown(
                    """
                    **Recommended actions:**

                    - 🤖 Use AI Tutor support
                    - 📅 Follow a manageable study schedule
                    - 📚 Start with easier learning resources
                    - 📊 Monitor learning activity regularly
                    """
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
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Personalized Learning Recommendation System | "
    "M.Tech Project"
)

