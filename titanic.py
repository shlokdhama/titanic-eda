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
plt.savefig('survival_by_class.png')
plt.show()
#first class passengers had a much higher survival rate than second and third class passengers

#age distrubution
plt.hist(df['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.savefig('age_distribution.png')

plt.show()
#most passengers were between 20-40 years old

#survival by age group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=['Child','Teen','Young Adult','Adult','Senior'])
survival_by_age = df.groupby('AgeGroup')['Survived'].mean()
plt.bar(survival_by_age.index, survival_by_age.values)
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_age_group.png')

plt.show()
# children had the highest survival rate, consistent with "women and children first"