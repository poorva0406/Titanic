import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

train_data = pd.read_csv('train.csv')
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

woman = train_data.loc[train_data.Sex == 'female']['Survived']
rate_woman = (sum(woman) / len(woman))*100
print('% of woman survived = ',rate_woman)

man = train_data.loc[train_data.Sex == 'male']['Survived']
rate_man = (sum(man) / len(man))*100
print('% of man survived = ',rate_man)

gender_data = pd.read_csv("~/Downloads/gender_submission.csv")
gender_data.head()

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
