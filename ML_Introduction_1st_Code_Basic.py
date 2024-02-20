#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
melbourne_file_path= 'https://raw.githubusercontent.com/ml008008/Repo1/main/melb_data%203.csv'
melbourne_data= pd.read_csv(melbourne_file_path)
melbourne_data.columns


# In[28]:


#not available , missing values
melbourne_data= melbourne_data.dropna(axis=0)


# In[29]:


y=melbourne_data.Price
y


# In[30]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X= melbourne_data[melbourne_features]
X


# In[31]:


X.describe()


# In[32]:


X.head()


# In[35]:


from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X,y)


# In[37]:


print("Making Prediction for the  following 5 houses: ")
print(X.head())
print("The predictions are")
print(melbourne_model.predict (X.head()))


# In[ ]:




