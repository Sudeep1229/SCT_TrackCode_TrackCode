
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load and Check the Dataset

df = pd.read_csv("train.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape (Rows, Columns):")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# 2. Data Cleaning


# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 3. Exploratory Data Analysis


# Survival Distribution
df['Survived'].value_counts().plot(kind='bar')
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Survival Distribution")
plt.show()

# Gender vs Survival
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Gender vs Survival")
plt.show()

# Passenger Class vs Survival
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar')
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.title("Passenger Class vs Survival")
plt.show()

# Age Distribution
plt.hist(df['Age'], bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Passengers")
plt.show()

# Fare vs Survival
df.boxplot(column='Fare', by='Survived')
plt.xlabel("Survived")
plt.ylabel("Fare")
plt.title("Fare vs Survival")
plt.suptitle("")
plt.show()


# 4. Family Size Analysis

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df.groupby('FamilySize')['Survived'].mean().plot(kind='bar')
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.title("Family Size vs Survival Rate")
plt.show()


# 5. Key Insights 

print("\nKEY INSIGHTS:")
print("1. Females had a higher survival rate than males.")
print("2. First-class passengers survived more than lower classes.")
print("3. Children had better survival chances.")
print("4. Higher fare passengers had higher survival probability.")
print("5. Small families survived more than solo or large families.")
