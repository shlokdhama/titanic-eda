import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

# preprocessing
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]

df = df[features + ["Survived"]]
df = df.dropna()

X = df[features]
y = df["Survived"]

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# save model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")