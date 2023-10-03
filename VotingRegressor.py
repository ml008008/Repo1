import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor

data = pd.read_csv('data_1.csv')

X = data[['Gender', 'Weight', 'Height']]
y = data['Shoe_Size']

models = [
    ('decision_tree', DecisionTreeRegressor()),
    ('linear_regression', LinearRegression()),
    ('k_neighbors', KNeighborsRegressor(n_neighbors=5))
]

model = VotingRegressor(models)
model.fit(X, y)

Weight = 50
Gender = 0 
Height = 158

Prediction = model.predict([[Gender, Weight, Height]])
print(f'If the weight is {Weight} kg, the Hight {Height} cm and the Person is a Man, Then the Shoe Sizes prediction would be  {Prediction[0]}')