# ============================================================
# PHASE 8 - DEPLOYMENT
# PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS - UI ONLY
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .info-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }

    .prediction-card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .prediction-label {
        font-size: 1rem;
        color: #666;
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #777;
        padding: 1rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "ml_model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


try:

    ml_model, encoder = load_models()

    model_loaded = True

except Exception as e:

    ml_model = None
    encoder = None
    model_loaded = False

    st.error(
        f"Unable to load the trained model: {str(e)}"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "feedback" not in st.session_state:

    st.session_state.feedback = []


if "generated_schedule" not in st.session_state:

    st.session_state.generated_schedule = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">

        <h1>🎓</h1>

        <h2>Personalized Learning</h2>

        <p>AI-Powered Learning Dashboard</p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("📌 Navigation")

    page = st.radio(
        "Choose a module",
        [
            "🏠 Dashboard",
            "📊 Performance Prediction",
            "📚 Course Recommendations",
            "🧠 Adaptive Learning",
            "📅 Study Schedule",
            "🤖 AI Tutor",
            "⭐ Feedback"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.subheader("System Status")

    if model_loaded:

        st.success("🟢 ML Model Loaded")

    else:

        st.error("🔴 ML Model Not Loaded")

    st.caption(
        "Personalized Learning Recommendation System"
    )

    st.caption(
        "M.Tech Project"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Personalized Learning Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered platform for personalized student learning, performance prediction, recommendations and scheduling.</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Student Learning Dashboard")

    st.write(
        """
        Welcome to the Personalized Learning Recommendation System.
        This platform analyzes student information and provides
        personalized learning support.
        """
    )

    st.markdown("### 📌 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "System Status",
            "Active"
        )

    with col2:

        if model_loaded:

            st.metric(
                "ML Model",
                "Ready"
            )

        else:

            st.metric(
                "ML Model",
                "Error"
            )

    with col3:

        st.metric(
            "Course Engine",
            "Available"
        )

    with col4:

        st.metric(
            "AI Tutor",
            "Available"
        )

    st.markdown("---")

    st.subheader("🔄 System Workflow")

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.markdown(
            """
            ### 1️⃣ Student Profile

            Student academic and learning information
            is provided.

            ### 2️⃣ Performance Prediction

            The trained machine learning model predicts
            the student's expected academic outcome.

            ### 3️⃣ Course Recommendation

            Relevant learning resources are recommended
            using the content-based recommendation engine.

            ### 4️⃣ Adaptive Learning

            Learning resources are organized according
            to the student's learning requirements.
            """
        )

    with workflow_col2:

        st.markdown(
            """
            ### 5️⃣ Study Scheduling

            A constraint-based scheduling algorithm creates
            a personalized study timetable.

            ### 6️⃣ AI Tutor

            Students can ask questions about important
            learning topics.

            ### 7️⃣ Feedback

            Students can provide ratings and feedback
            about the system.

            ### 🎯 Goal

            Provide a personalized, adaptive and
            student-friendly learning experience.
            """
        )

    st.markdown("---")

    st.subheader("🚀 Quick Start")

    quick1, quick2, quick3 = st.columns(3)

    with quick1:

        st.info(
            """
            **📊 Predict Performance**

            Enter student information and obtain
            the predicted academic outcome.
            """
        )

    with quick2:

        st.info(
            """
            **📚 Explore Recommendations**

            View personalized learning resources
            generated by the recommendation engine.
            """
        )

    with quick3:

        st.info(
            """
            **📅 Create Study Schedule**

            Select your available days and times
            to generate a personalized timetable.
            """
        )


# ============================================================
# PERFORMANCE PREDICTION
# ============================================================

elif page == "📊 Performance Prediction":

    st.header("📊 Student Performance Prediction")

    st.write(
        """
        Enter the student's academic and learning information.
        The trained machine learning model will predict the
        student's expected outcome.
        """
    )

    if not model_loaded:

        st.error(
            "The trained model could not be loaded. "
            "Please verify that ml_model.pkl and encoder.pkl "
            "are present in the application folder."
        )

        st.stop()

    # ========================================================
    # USER-FRIENDLY CATEGORY MAPPINGS
    # ========================================================

    gender_map = {
        "♀ Female": 0,
        "♂ Male": 1
    }

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

    education_map = {
        "A Level or Equivalent": 0,
        "HE Qualification": 1,
        "Lower Than A Level": 2,
        "No Formal Qualification": 3,
        "Post Graduate Qualification": 4
    }

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

    disability_map = {
        "No": 0,
        "Yes": 1
    }

    # ========================================================
    # SECTION 1 - PERSONAL INFORMATION
    # ========================================================

    st.subheader("👤 1. Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender_label = st.radio(
            "Gender",
            options=[
                "♀ Female",
                "♂ Male"
            ],
            horizontal=True
        )

        gender = gender_map[gender_label]

        age = st.selectbox(
            "Age",
            options=list(range(18, 71)),
            index=2
        )

        if age < 35:

            age_band = 0

        elif age < 55:

            age_band = 1

        else:

            age_band = 2

    with col2:

        region_label = st.selectbox(
            "Region",
            options=list(region_map.keys())
        )

        region = region_map[region_label]

        education_label = st.selectbox(
            "Highest Education",
            options=list(education_map.keys())
        )

        highest_education = education_map[
            education_label
        ]

    with col3:

        imd_label = st.selectbox(
            "IMD Band",
            options=list(imd_map.keys())
        )

        imd_band = imd_map[imd_label]

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
    # AGE INFORMATION
    # ========================================================

    with st.expander("ℹ️ About Age Band"):

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

        st.caption(
            "The system automatically converts the selected age "
            "into the age-band representation used by the model."
        )

    st.markdown("---")

    # ========================================================
    # SECTION 2 - ACADEMIC INFORMATION
    # ========================================================

    st.subheader("📚 2. Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        num_of_prev_attempts = st.number_input(
            "Previous Attempts",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Number of previous attempts made by the student."
        )

        studied_credits = st.number_input(
            "Studied Credits",
            min_value=0.0,
            value=60.0,
            step=10.0,
            help="Number of credits studied by the student."
        )

    with col2:

        total_assessments = st.number_input(
            "Total Assessments",
            min_value=0.0,
            value=5.0,
            step=1.0,
            help="Total number of assessments."
        )

        average_score = st.number_input(
            "Average Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Average assessment score."
        )

    with col3:

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

    st.markdown("---")

    # ========================================================
    # SECTION 3 - LEARNING ACTIVITY
    # ========================================================

    st.subheader("📈 3. Learning Activity")

    col1, col2, col3 = st.columns(3)

    with col1:

        total_clicks = st.number_input(
            "Total Clicks",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Total number of interactions/clicks."
        )

        average_clicks = st.number_input(
            "Average Clicks",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

    with col2:

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

    with col3:

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

    st.markdown("---")

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("🔎 Review Student Information"):

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(
                f"**Gender:** {gender_label}"
            )

            st.write(
                f"**Age:** {age} years"
            )

            st.write(
                f"**Region:** {region_label}"
            )

            st.write(
                f"**Education:** {education_label}"
            )

            st.write(
                f"**IMD Band:** {imd_label}"
            )

        with summary_col2:

            st.write(
                f"**Previous Attempts:** {num_of_prev_attempts}"
            )

            st.write(
                f"**Studied Credits:** {studied_credits}"
            )

            st.write(
                f"**Average Score:** {average_score}"
            )

            st.write(
                f"**Active Days:** {active_days}"
            )

            st.write(
                f"**Course Duration:** {course_duration}"
            )

    # ========================================================
    # PREDICTION
    # ========================================================

    st.markdown("### 🎯 Generate Prediction")

    if st.button(
        "🔍 Predict Student Performance",
        type="primary",
        use_container_width=True
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

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        Predicted Student Outcome
                    </div>

                    <div class="prediction-value">
                        {result}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            if result == "Fail":

                st.warning(
                    "⚠️ The prediction indicates that the "
                    "student may require additional learning support."
                )

                st.subheader("💡 Recommended Actions")

                action_col1, action_col2 = st.columns(2)

                with action_col1:

                    st.markdown(
                        """
                        - 📚 Review weak topics
                        - 📝 Increase practice sessions
                        - ❓ Take frequent quizzes
                        """
                    )

                with action_col2:

                    st.markdown(
                        """
                        - 🤖 Use AI Tutor assistance
                        - 📅 Follow the personalized schedule
                        - 📊 Monitor learning progress
                        """
                    )

            elif result == "Pass":

                st.success(
                    "✅ The student is progressing satisfactorily."
                )

                st.subheader("💡 Recommended Actions")

                st.markdown(
                    """
                    - Continue the current learning plan
                    - Practice regularly
                    - Follow the recommended study schedule
                    - Use quizzes to maintain performance
                    """
                )

            elif result == "Distinction":

                st.success(
                    "🏆 Excellent performance! "
                    "Advanced learning resources are recommended."
                )

                st.subheader("💡 Recommended Actions")

                st.markdown(
                    """
                    - 🚀 Explore advanced topics
                    - 📚 Attempt challenging resources
                    - 🧠 Practice advanced problems
                    - 🎯 Maintain consistent learning
                    """
                )

            elif result == "Withdrawn":

                st.warning(
                    "⚠️ The student may be at risk of disengagement."
                )

                st.subheader("💡 Recommended Actions")

                st.markdown(
                    """
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

elif page == "📚 Course Recommendations":

    st.header("📚 Personalized Course Recommendations")

    st.write(
        """
        This section displays the learning resources generated
        by the Phase 3 Content-Based Recommendation Engine.
        """
    )

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

        st.success(
            f"✓ {len(recommendations)} recommendation(s) available."
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:

            st.metric(
                "Learning Resources",
                len(recommendations)
            )

        with metric_col2:

            if "difficulty" in recommendations.columns:

                difficulty_count = (
                    recommendations["difficulty"]
                    .nunique()
                )

                st.metric(
                    "Difficulty Levels",
                    difficulty_count
                )

            else:

                st.metric(
                    "Difficulty Levels",
                    "Available"
                )

        with metric_col3:

            if "Similarity" in recommendations.columns:

                avg_similarity = (
                    recommendations["Similarity"]
                    .mean()
                )

                st.metric(
                    "Avg. Similarity",
                    f"{avg_similarity:.2f}"
                )

            else:

                st.metric(
                    "Recommendation Engine",
                    "Active"
                )

        st.markdown("---")

        st.subheader("📋 Recommended Learning Resources")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="⬇️ Download Recommendations",
            data=recommendations.to_csv(
                index=False
            ),
            file_name="personalized_recommendations.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "recommendations.csv was not found."
        )

        st.info(
            """
            Please export the Phase 3 recommendation DataFrame
            using:

            recommendations.to_csv(
                "recommendations.csv",
                index=False
            )
            """
        )


# ============================================================
# ADAPTIVE LEARNING
# ============================================================

elif page == "🧠 Adaptive Learning":

    st.header("🧠 Adaptive Learning Plan")

    st.write(
        """
        The adaptive learning module organizes recommended
        resources according to the student's learning requirements.
        """
    )

    adaptive_file = BASE_DIR / "adaptive_plan.csv"

    if adaptive_file.exists():

        adaptive_plan = pd.read_csv(
            adaptive_file
        )

        st.success(
            "✓ Adaptive learning plan loaded successfully."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Learning Resources",
                len(adaptive_plan)
            )

        with col2:

            if "Learning Level" in adaptive_plan.columns:

                levels = (
                    adaptive_plan["Learning Level"]
                    .nunique()
                )

                st.metric(
                    "Learning Levels",
                    levels
                )

            else:

                st.metric(
                    "Learning Plan",
                    "Ready"
                )

        with col3:

            if "Estimated Time" in adaptive_plan.columns:

                st.metric(
                    "Estimated Time",
                    "Available"
                )

            else:

                st.metric(
                    "Adaptive Engine",
                    "Active"
                )

        st.markdown("---")

        st.subheader("📋 Adaptive Learning Resources")

        st.dataframe(
            adaptive_plan,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="⬇️ Download Adaptive Learning Plan",
            data=adaptive_plan.to_csv(
                index=False
            ),
            file_name="adaptive_learning_plan.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "adaptive_plan.csv was not found."
        )

        st.info(
            """
            Please export the Phase 4 result using:

            adaptive_plan.to_csv(
                "adaptive_plan.csv",
                index=False
            )
            """
        )


# ============================================================
# STUDY SCHEDULE
# ============================================================

elif page == "📅 Study Schedule":

    st.header("📅 Personalized Study Schedule")

    st.write(
        """
        Select the days and time range when you are available
        for studying. The system will generate a personalized
        schedule based on your adaptive learning plan.
        """
    )

    adaptive_file = BASE_DIR / "adaptive_plan.csv"

    if not adaptive_file.exists():

        st.warning(
            "adaptive_plan.csv was not found. "
            "Please generate/export the Phase 4 adaptive learning plan first."
        )

        st.stop()

    adaptive_plan = pd.read_csv(
        adaptive_file
    )

    # ========================================================
    # STEP 1
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
        default=[
            "Monday",
            "Wednesday",
            "Friday"
        ]
    )

    # ========================================================
    # STEP 2
    # ========================================================

    st.subheader("2️⃣ Select Available Study Time")

    col1, col2 = st.columns(2)

    with col1:

        study_start_time = st.time_input(
            "Study Start Time",
            value=pd.to_datetime(
                "18:00"
            ).time()
        )

    with col2:

        study_end_time = st.time_input(
            "Study End Time",
            value=pd.to_datetime(
                "21:00"
            ).time()
        )

    # ========================================================
    # STEP 3
    # ========================================================

    if selected_days:

        st.subheader("3️⃣ Your Availability")

        availability_col1, availability_col2 = st.columns(2)

        with availability_col1:

            st.info(
                f"📅 **Days:** "
                f"{', '.join(selected_days)}"
            )

        with availability_col2:

            st.info(
                f"⏰ **Time:** "
                f"{study_start_time.strftime('%I:%M %p')} - "
                f"{study_end_time.strftime('%I:%M %p')}"
            )

    # ========================================================
    # STEP 4
    # ========================================================

    st.subheader("4️⃣ Generate Schedule")

    if st.button(
        "📅 Generate Personalized Schedule",
        type="primary",
        use_container_width=True
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

            start_hour = (
                study_start_time.hour
                + study_start_time.minute / 60
            )

            end_hour = (
                study_end_time.hour
                + study_end_time.minute / 60
            )

            start_slot = int(
                start_hour * 4
            )

            end_slot = int(
                end_hour * 4
            )

            try:

                schedule_rows = []

                available_minutes_per_day = (
                    end_hour - start_hour
                ) * 60

                day_index = 0

                current_day = selected_days[
                    day_index
                ]

                current_slot = start_slot

                for _, row in adaptive_plan.iterrows():

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

                    if (
                        current_slot
                        + required_slots
                        > end_slot
                    ):

                        day_index += 1

                        if day_index >= len(
                            selected_days
                        ):

                            day_index = 0

                        current_day = selected_days[
                            day_index
                        ]

                        current_slot = start_slot

                    start_minutes = (
                        current_slot * 15
                    )

                    end_minutes = (
                        (
                            current_slot
                            + required_slots
                        )
                        * 15
                    )

                    start_h = (
                        start_minutes // 60
                    )

                    start_m = (
                        start_minutes % 60
                    )

                    end_h = (
                        end_minutes // 60
                    )

                    end_m = (
                        end_minutes % 60
                    )

                    start_time = (
                        f"{int(start_h):02d}:"
                        f"{int(start_m):02d}"
                    )

                    end_time = (
                        f"{int(end_h):02d}:"
                        f"{int(end_m):02d}"
                    )

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

                    current_slot += (
                        required_slots
                    )

                generated_schedule = pd.DataFrame(
                    schedule_rows
                )

                st.session_state.generated_schedule = (
                    generated_schedule
                )

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

                st.download_button(
                    label="⬇️ Download Study Schedule",
                    data=generated_schedule.to_csv(
                        index=False
                    ),
                    file_name="personalized_study_schedule.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Schedule generation error: {str(e)}"
                )

    # ========================================================
    # PREVIOUSLY GENERATED SCHEDULE
    # ========================================================

    if (
        st.session_state.generated_schedule
        is not None
    ):

        st.markdown("---")

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

elif page == "🤖 AI Tutor":

    st.header("🤖 AI Tutor")

    st.write(
        """
        Ask questions about important machine learning,
        deep learning and artificial intelligence topics.
        """
    )

    st.info(
        """
        💡 **Example questions**

        • Explain CNN in simple terms.

        • What is LSTM?

        • What is Machine Learning?

        • Explain neural networks.
        """
    )

    question = st.text_area(
        "💬 Enter your question",
        placeholder="Example: Explain CNN in simple terms.",
        height=150
    )

    if st.button(
        "🤖 Ask AI Tutor",
        type="primary"
    ):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

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

            st.markdown("---")

            st.subheader(
                "💡 AI Tutor Response"
            )

            st.success(
                "Your question has been processed."
            )

            st.write(
                response
            )


# ============================================================
# FEEDBACK
# ============================================================

elif page == "⭐ Feedback":

    st.header("⭐ Student Feedback")

    st.write(
        """
        Your feedback helps evaluate and improve the
        Personalized Learning Recommendation System.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        rating = st.slider(
            "⭐ Rate the system",
            min_value=1,
            max_value=5,
            value=5
        )

    with col2:

        st.write(
            f"### Your Rating: {rating} / 5"
        )

        if rating == 5:

            st.write("Excellent! 🎉")

        elif rating == 4:

            st.write("Very Good! 👍")

        elif rating == 3:

            st.write("Good. 🙂")

        elif rating == 2:

            st.write("Needs Improvement. 📝")

        else:

            st.write(
                "We appreciate your feedback. 🙏"
            )

    feedback = st.text_area(
        "📝 Enter your feedback",
        placeholder=(
            "Tell us about your experience with "
            "the Personalized Learning System..."
        ),
        height=150
    )

    if st.button(
        "⭐ Submit Feedback",
        type="primary"
    ):

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

    if len(
        st.session_state.feedback
    ) > 0:

        st.markdown("---")

        st.subheader(
            "📋 Feedback Records"
        )

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

st.markdown(
    """
    <div class="footer">

    🎓 <b>Personalized Learning Recommendation System</b>
    <br>
    AI-Powered Adaptive Learning Platform
    <br>
    M.Tech Project

    </div>
    """,
    unsafe_allow_html=True
)
