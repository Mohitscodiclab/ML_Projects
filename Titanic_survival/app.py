import numpy as np
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create directory structure for saving visualizations
vis_dir = 'Visualization'
imgs_dir = os.path.join(vis_dir, 'imgs')
os.makedirs(imgs_dir, exist_ok=True)  # Create directory if it doesn't exist

# Function to handle plot closing and saving
def save_on_close(fig, filename):
    def on_close(event):
        if not os.path.exists(filename):
            fig.savefig(filename)
            print(f"Saved visualization: {filename}")
        else:
            print(f"Visualization already exists: {filename}")
    fig.canvas.mpl_connect('close_event', on_close)

# Load dataset from csv file to Pandas DataFrame
titanic_data = pd.read_csv('train.csv')

# Print the first 5 rows of the dataframe
print(titanic_data.head())
# sleep(1)
# Number of rows and columns in the dataframe
print(titanic_data.shape)
# sleep(1)
# Getting some information about the data
print(titanic_data.info())
sleep(1)
# Check the missing values in the dataset
print(titanic_data.isnull().sum())
sleep(1)
# Handling the missing values in the dataset
# Dropping the 'Cabin' column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
print("After dropping Cabin column:")
print(titanic_data.head())
sleep(1)
# Replacing the missing values in the 'Age' column with the mean age
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
# finding the mode value of 'Embarked' column
print("Mode value of Embarked column:", titanic_data['Embarked'].mode())
sleep(2)
# Replacing the missing values in the 'Embarked' column with the mode value
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
# Check missing values again after handling
print("\nMissing values after handling:")
print(titanic_data.isnull().sum())
sleep(2)
#getting some statistical measures about the data
print(titanic_data.describe())
# finding the number of survived and non survived passengers
print(titanic_data['Survived'].value_counts()) 









# Data visualization
sns.set()

# Create a figure for the plot
plt.figure()

# Make a count plot for 'Survived' column
sns.countplot(x='Survived', data=titanic_data)

# Set plot title and labels
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.xlabel('Survived passengers Overall')
plt.ylabel('Count')

# Save the plot using our custom function
save_on_close(plt.gcf(), os.path.join(imgs_dir, 'survived_countplot.png'))

# Display the plot
plt.show()








sns.countplot(x='Sex', data=titanic_data)

# Set plot title and labels
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.xlabel('Survived overall Male and Female')
plt.ylabel('Count')

# Save the plot using our custom function
save_on_close(plt.gcf(), os.path.join(imgs_dir, 'survived_by_sex.png'))

# Display the plot
plt.show()


# Finding the number of Sex no of male and female passengers survived 
print(titanic_data['Sex'].value_counts())
# total male passengers survived = 577
# total female passengers survived = 314







# number of survivors based on gender
sns.countplot(x='Sex', hue='Survived', data=titanic_data)

# Set plot title and labels
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.xlabel('Survived passengers Gender Based')
plt.ylabel('Count')

# Save the plot using our custom function
save_on_close(plt.gcf(), os.path.join(imgs_dir, 'survived_by_Gender.png'))

# Display the plot
plt.show()








# number of survivors based on gender
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

# Set plot title and labels
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.xlabel('Survived Based on Pclass')
plt.ylabel('Count')

# Save the plot using our custom function
save_on_close(plt.gcf(), os.path.join(imgs_dir, 'survived_by_Pclass.png'))

# Display the plot
plt.show()





print(titanic_data['Survived'].value_counts())
# 549 passengers did not survive
# 342 passengers survived



# Encoding the categorical columns
print(titanic_data['Sex'].value_counts())


print(titanic_data['Embarked'].value_counts())


# converting the catogorical columns into numerical columns
# Now we will replace male with 0 and female with 1 in the 'dataframe'
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(titanic_data.head())

# separating the features and target
x = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1) 
y = titanic_data['Survived']

print(x)
print(y)
# splitting the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape) 

# training the model
model = LogisticRegression()
# training the LogisticRegression model with training data
model.fit(x_train, y_train)



# accuracy on training data
x_train_prediction = model.predict(x_train)
print(x_train_prediction)
training_data_accuracy = accuracy_score(y_train, x_train_prediction) 
print("Accuracy on training data : ", training_data_accuracy) 



# Accuracy on test data
x_test_prediction = model.predict(x_test)
print(x_test_prediction)

# Accuracy on test data
test_data_accuracy = accuracy_score(y_test, x_test_prediction) 
print("Accuracy on test data : ", test_data_accuracy) 

 





# Now we will make a prediction system
def get_passenger_input():
    """Get passenger details from user input"""
    print("\n=== Titanic Survival Prediction ===")
    print("Please enter the passenger details:")
    
    # Get Pclass (1, 2, or 3)
    while True:
        try:
            pclass = int(input("Passenger class (1, 2, or 3): "))
            if pclass in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Get Sex (male or female)
    while True:
        sex = input("Sex (male/female): ").lower()
        if sex in ['male', 'female']:
            sex = 0 if sex == 'male' else 1
            break
        else:
            print("Please enter 'male' or 'female'")
    
    # Get Age
    while True:
        try:
            age = float(input("Age: "))
            if 0 < age < 120:
                break
            else:
                print("Please enter a valid age between 0 and 120")
        except ValueError:
            print("Please enter a valid number")
    
    # Get SibSp (number of siblings/spouses)
    while True:
        try:
            sibsp = int(input("Number of siblings/spouses aboard: "))
            if sibsp >= 0:
                break
            else:
                print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get Parch (number of parents/children)
    while True:
        try:
            parch = int(input("Number of parents/children aboard: "))
            if parch >= 0:
                break
            else:
                print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get Fare
    while True:
        try:
            fare = float(input("Ticket fare: "))
            if fare >= 0:
                break
            else:
                print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get Embarked (S, C, or Q)
    while True:
        embarked = input("Port of Embarkation (S=Southampton, C=Cherbourg, Q=Queenstown): ").upper()
        if embarked in ['S', 'C', 'Q']:
            embarked = 0 if embarked == 'S' else (1 if embarked == 'C' else 2)
            break
        else:
            print("Please enter S, C, or Q")
    
    return np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

def predict_survival():
    """Predict survival based on user input"""
    # Get passenger details
    passenger_data = get_passenger_input()
    
    # Make prediction
    prediction = model.predict(passenger_data)
    prediction_proba = model.predict_proba(passenger_data)
    
    # Display results
    print("\n=== Prediction Results ===")
    if prediction[0] == 1:
        print("Result: This passenger would SURVIVE")
    else:
        print("Result: This passenger would NOT survive")
    
    print(f"Survival probability: {prediction_proba[0][1]:.2%}")
    print(f"Death probability: {prediction_proba[0][0]:.2%}")
    
    # Feature importance (coefficients)
    print("\n=== Feature Importance ===")
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    coefficients = model.coef_[0]
    
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef:.4f}")

# Run the prediction
predict_survival()