import os
print(os.listdir())  # List all files in the current directory
file_path = os.path.join("C:\\Users\\aryan\\OneDrive\\Documents\\phython project\\Titanic - Machine Learning from Disaster", "train.csv")#join path 

import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt  
df = pd.read_csv(file_path)
# print(df.head())
# sns.boxplot(df['Age'])  # finding outliers 
# plt.show()

# finding missing value...
df['Age']=df['Age'].fillna(df["Age"].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)
df.drop(columns=['Ticket'], inplace=True)
df.drop(columns=['Name'], inplace=True)

# Text Data Ko Numbers Me Convert Karna...
df['Sex']=df['Sex'].map({"male":1,"female":0})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Features Aur Target Column Select Karna...
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]#x is used basis for model predict
y = df['Survived']# y ued for predict 
print(df.head())

#MODEL TRAINING 
#Train-Test Split(Traning and testing )
from sklearn.model_selection import train_test_split

# Dataset ko split karna (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model select and train 
from sklearn.ensemble import RandomForestClassifier

# Model initialize karo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model ko training data par fit karo
from sklearn.ensemble import RandomForestClassifier

# Model initialize karo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model ko training data par fit karo
from sklearn.ensemble import RandomForestClassifier

# Model initialize karo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model ko training data par fit karo
from sklearn.ensemble import RandomForestClassifier
# Model initialize 
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Model ko training data par fit karo
model.fit(X_train, y_train)
#model accuracy check
from sklearn.metrics import accuracy_score

# Model prediction karega test data ke liye
y_pred = model.predict(X_test)
print(y_pred)
# Accuracy check karna
from sklearn.metrics import accuracy_score
# Model prediction for  test data 
y_pred = model.predict(X_test)
 # Accuracy check 
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
