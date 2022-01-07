
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv('C:/Users/Dell/Downloads/Iris.csv')
print(data.shape)

print(data.columns)

print(data.head)

print(data.tail)

print(data.describe())

print(data.isnull().any())
#%%

sns.boxplot(y=data['SepalLengthCm'])
#%%

sns.boxplot(y=data['SepalWidthCm'])
#%%
sns.boxplot(y=data['PetalLengthCm'])
#%%
sns.boxplot(y=data['PetalWidthCm'])


#%%

features = data.drop('Species', axis=1)
target = data['Species']

scale = StandardScaler()

scale.fit(features)

scaled_features=scale.transform(features)

data_new = pd.DataFrame(scaled_features)
data_new.head(3)
#%%

x_train, x_test, y_train, y_test = train_test_split(data_new, target, test_size=0.25, random_state=45)


x_train.shape

x_train.head()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(pred)

print(classification_report(y_test, pred))

accuracy = model.score(x_test, y_test)
print(accuracy*100,'%')