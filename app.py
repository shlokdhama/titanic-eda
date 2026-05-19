import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "titanic_model.pkl")

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

with open(model_path, "rb") as file:
    model = pickle.load(file)

# =========================================================
# PAGE TITLE
# =========================================================

st.title("Titanic Survival Predictor")

st.write(
    "This machine learning app predicts whether a passenger "
    "would likely survive the Titanic disaster."
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("About")

st.sidebar.write(
    """
    Built using:
    - Logistic Regression
    - Scikit-learn
    - Streamlit
    - Titanic Dataset from Kaggle
    """
)

# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

fare = st.slider(
    "Fare",
    0,
    600,
    50
)

sibsp = st.number_input(
    "Siblings/Spouses aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents/Children aboard",
    min_value=0,
    max_value=10,
    value=0
)

# =========================================================
# PREPROCESS INPUT
# =========================================================

sex = 0 if sex == "Male" else 1

input_data = pd.DataFrame(
    [[pclass, sex, age, fare, sibsp, parch]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "SibSp",
        "Parch"
    ]
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The model predicts that the passenger would likely survive."
        )
    else:
        st.error(
            "The model predicts that the passenger would likely not survive."
        )

    st.write(f"Survival Probability: {probability:.2%}")

    st.write("Passenger Data:")

    st.dataframe(input_data)