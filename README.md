# Titanic Survival Prediction

An end-to-end machine learning classification project built using the Kaggle Titanic dataset.  
The project covers data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and deployment using Streamlit.

## Live Demo

Streamlit App:  
https://titanic-eda-shlok.streamlit.app

## Project Overview

The goal of this project was to predict whether a passenger survived the Titanic disaster using features such as passenger class, sex, age, fare, and family relationships onboard.

I trained and compared multiple classification models and deployed the final model as an interactive web app using Streamlit.

## Key Insight

The model identified sex as the strongest predictor of survival. Female passengers had a significantly higher chance of survival, reflecting the historical "women and children first" evacuation policy.

Passenger class was also strongly linked to survival probability, showing how social and economic status affected access to lifeboats.

## App Screenshots

### Main Interface
<img width="1857" height="877" alt="image" src="https://github.com/user-attachments/assets/bf02fdef-728a-404f-84e4-80c29f950081" />


### Prediction Example
<img width="620" height="775" alt="image" src="https://github.com/user-attachments/assets/42e4c0e8-967f-44b7-ab39-150c4461f2b0" />
<img width="602" height="791" alt="image" src="https://github.com/user-attachments/assets/b48676c8-67c4-43ae-8f5f-432bbfef6bec" />


## What I Did

### Data Cleaning
- Filled missing values in the `Age` column using the median
- Dropped the `Cabin` column because most values were missing
- Selected the most useful features for prediction

### Exploratory Data Analysis
- Analysed survival trends across passenger class, sex, and age
- Visualised distributions and correlations using matplotlib and seaborn

### Feature Engineering
- Encoded categorical features such as `Sex`
- Prepared the dataset for machine learning models

### Model Training
Trained and compared:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

### Model Evaluation
Evaluated models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 81.0% |
| Random Forest | 79.3% |
| Decision Tree | 77.7% |

**Best Performing Model:** Logistic Regression

## Logistic Regression Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Did not survive | 0.82 | 0.87 | 0.84 |
| Survived | 0.79 | 0.73 | 0.76 |

## What the Model Learned

- `Sex` was the strongest predictor of survival
- Lower passenger classes had lower survival rates
- Age had a weaker independent effect after accounting for sex and class

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Streamlit

## Project Structure

```text
titanic/
│
├── app.py
├── model.pkl
├── train.csv
├── requirements.txt
├── README.md
└── titanic.py
```

## Dataset

Kaggle Titanic Competition:
https://www.kaggle.com/competitions/titanic
