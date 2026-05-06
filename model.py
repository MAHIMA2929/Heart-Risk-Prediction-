import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# dataset load
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# missing values remove
data = data.dropna()

# features
X = data[['age','hypertension','heart_disease','avg_glucose_level','bmi']]
y = data['stroke']

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model
model = LogisticRegression(max_iter=1000)

# train
model.fit(X_train,y_train)

# save model
pickle.dump(model, open("model.pkl","wb"))

print("Model trained successfully")
