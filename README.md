
# Loan Prediction Model

This repository contains code for a loan prediction model. The model utilizes the Support Vector Machine (SVM) algorithm to predict the approval status of loan applications based on various features.

## Prerequisites

Before running the code, ensure that the following libraries are installed:

- numpy 📦
- pandas 📦
- seaborn 📦
- scikit-learn 📦

## Dataset

I used the [Loan Predication Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication) from Kaggle. The dataset provides valuable information to train the loan prediction model. It is available in CSV format and can be loaded into a pandas DataFrame using the `read_csv()` function. The dataset should be located at the specified file path.

```python
# loading the dataset to pandas DataFrame
import pandas as pd

df = pd.read_csv('/path/to/dataset.csv')
df.head()
df.describe()
```

## Data Preprocessing

The dataset undergoes preprocessing to handle missing values and convert categorical columns into numerical values. The following steps are performed:

- Checking the number of missing values in each column using `isnull().sum()` ❓
- Dropping the rows with missing values using `dropna()` ❌
- Replacing the value '3+' in the 'Dependents' column with '4' 🔢
- Converting categorical columns ('Gender', 'Education', 'Property_Area', 'Self_Employed', 'Married') to numerical values using `replace()` ♻️

## Model Training and Evaluation

The data is split into training and testing sets using `train_test_split()` from scikit-learn. The SVM model with a linear kernel is trained using the training set (`svm.SVC(kernel='linear')`).

The accuracy of the model is evaluated on both the training and testing data using the `accuracy_score()` function from scikit-learn. The accuracy scores are printed for both the training and testing data.

```python
# separating the data and label
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = df['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)
```

Accuracy score on training data
```python
x_train_prediction = classifier.predict(x_train)
train_data_acc = accuracy_score(x_train_prediction, y_train)
print(f'Accuracy on training data: {train_data_acc}')
```

Accuracy score on testing data
```python
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(f'Accuracy on testing data: {test_data_accuracy}')
```

## Usage

To use this code:

1. Ensure that the dataset is in CSV format.
2. Update the file path in the line `df = pd.read_csv('/path/to/dataset.csv')` to the correct location of your dataset.
3. Install the required libraries if not already installed.
4. Run the code and observe the accuracy scores on the training and testing data.

🚨 Note: This code is provided as a prototype and may require further enhancements and optimizations for real-world scenarios.

📝 Feel free to copy the code directly to your GitHub repository and modify it to suit your specific requirements.
