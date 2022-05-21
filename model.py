import numpy as np
import pandas as pd

df=pd.read_csv('breast_cancer.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())

x=df.drop('diagnosis',axis=1)
y=df.diagnosis

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100,shuffle=True)

from sklearn.ensemble import ExtraTreesClassifier
et_model=ExtraTreesClassifier()

et_model.fit(x_train,y_train)

prediction= et_model.predict(x_test)

import pickle
pickle.dump(et_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
