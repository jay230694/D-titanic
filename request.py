# -*- coding: utf-8 -*-
 

  
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'PassengerId':810,'Pclass':1,'Sex':1,'Age':33.0,'SibSp':1,'Parch':0,'Fare':53.1000})

print(r.json())
