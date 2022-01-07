 
import pandas as pd
import numpy as np
import pickle

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("C:/Users/IDRIS/Downloads/tested.csv")

df.shape

data = pd.read_csv("C:/Users/IDRIS/Downloads/tested.csv")

data.shape

data = data.dropna()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = data.drop(['Embarked', 'Name', 'Cabin', 'Ticket'], axis = 1)

data['Sex'] = data['Sex'].map({'male': 0,'female': 1})

X_train, X_test, y_train, y_test = train_test_split(data.drop(['Survived'],axis=1), 
                                                    data['Survived'], test_size=0.20, 
                                                    random_state=8)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


 
#Saving model to disk
pickle.dump(logmodel, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[810,1,1,33.0,1,0,53.1000]]))

