import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the dataset to pandas DataFrame
df = pd.read_csv('/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')
df.head()
df.describe()

# number of missing values in each column
df.isnull().sum()

# dropping the missing values
df = df.dropna()
df.isnull().sum()
df.head()

# Dependent column values
df['Dependents'].value_counts()

# replacing the value of 3+ to 4
df = df.replace(to_replace='3+', value=4)
df['Dependents'].value_counts()

# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=df)

# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data)
df.head(10)

# convert categorical columns to numerical values
df.replace({"Gender":{'Male':1,'Female':0},
           "Education":{'Graduate':1,'Not Graduate':0},
           "Property_Area":{'Rural':0,'Semiurban':1,'Urban':2},
           "Self_Employed":{'Yes':1,'No':0},
           "Married":{'Yes':1,'No':0}},inplace=True)
           
df.head(10)
# separating the data and label
x = df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = df['Loan_Status']

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=42)
classifier= svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

# accuracy score on test data
x_train_prediction = classifier.predict(x_train)
train_data_acc = accuracy_score(x_train_prediction,y_train)
print(f'Accuracy on training data : {train_data_acc}')


# accuracy score on training data
x_test_prediction = classifier.predict(x_test)
test_data_accuray = accuracy_score(x_test_prediction,y_test)
print(f'Accuracy on training data : {test_data_accuray}')
