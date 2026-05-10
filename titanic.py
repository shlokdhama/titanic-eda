import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('titanic/train.csv')

#EDA - exploratory data analysis

print(df.head()) #print first 5 rows
print(df.shape) #how many row and columns
print(df.describe())  #general stats
print(df.isnull().sum()) #which columns have missing values

df.drop(['Cabin', 'Ticket'], axis=1, inplace=True) #drop columns with too many missing values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) 
df['Age'] = df['Age'].fillna(df['Age'].median()) #fill missing age values with median

print(df.isnull().sum()) #updated missing values check

#visualization of survival counts
survival_counts = df['Survived'].value_counts()
plt.bar(['Did not Survive', 'Survived'], survival_counts)
plt.title('Survival Counts')
plt.ylabel('Number of Passengers')
plt.show()

#survival by gender
survival_by_gender = df.groupby('Sex')['Survived'].mean()
plt.bar(survival_by_gender.index, survival_by_gender.values)
plt.title('Survival by Gender')
plt.ylabel('Survival Rate')

plt.show()

#the survival rate of women is much higher than men

#survival by passenger class
survival_by_class = df.groupby('Pclass')['Survived'].mean()
plt.bar(survival_by_class.index, survival_by_class.values)
plt.title('Survival by Passenger Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.show()
#first class passengers had a much higher survival rate than second and third class passengers

#age distrubution
plt.hist(df['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.show()
#most passengers were between 20-40 years old

#survival by age group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=['Child','Teen','Young Adult','Adult','Senior'])
survival_by_age = df.groupby('AgeGroup')['Survived'].mean()
plt.bar(survival_by_age.index, survival_by_age.values)
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')

plt.show()
# children had the highest survival rate, consistent with "women and children first"


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

#prepare data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

#pick features and target
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

#split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training model
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_predictions = lr.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#random forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

#decision tree model
df = DecisionTreeClassifier(random_state=42)
df.fit(X_train, y_train)
dt_predictions = df.predict(X_test)

#comparing
print("Logistic Regression:", accuracy_score(y_test, lr_predictions))
print("Random Forest:", accuracy_score(y_test, rf_predictions))
print("Decision Tree:", accuracy_score(y_test, dt_predictions))

#confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, lr_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did not survive', 'Survived'])
disp.plot()
plt.title('Logistic Regression — Confusion Matrix')
plt.savefig('logistic_regression_confusion_matrix.png', dpi=300)
plt.show()

#feature importance
import numpy as np

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_[0]
})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

plt.bar(coef_df['Feature'], coef_df['Coefficient'])
plt.title('Feature Coefficients — Logistic Regression')
plt.ylabel('Coefficient Value')
plt.savefig('logistic_regression_feature_importance.png', dpi=300)
plt.show()

#classification report
print(classification_report(y_test, lr_predictions, target_names=['Did not survive', 'Survived']))