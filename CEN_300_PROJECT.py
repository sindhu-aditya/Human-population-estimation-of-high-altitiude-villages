#!/usr/bin/env python
# coding: utf-8

# ## 1)DATA PROCESSING

# In[1]:


#including lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #for visualisation-matlab and seaborn
import seaborn as sns


# ### 1.1)IMPORTING DATABASE

# In[2]:


#IMPORTING DATABASE
dataset=pd.read_csv('Villages List - Sheet1 (3).csv')
dataset.head()


# In[3]:


dataset.info()


# ### 1.2) CLEANING OF DATA

# In[4]:


#we will drop village name as it a categorical data and encode location
dataset=dataset.drop("Village Name",axis=1)
dataset=pd.get_dummies(dataset,columns=['Location'])
dataset.head()


# By looking at our dataset we observe that only Population can be treated as dependent varaible. So lets shift it to the last column.

# In[5]:


def reorder_columns(dataframe, col_name, position):
    temp_col = dataframe[col_name]
    dataframe = dataframe.drop(columns=[col_name])
    dataframe.insert(loc=position, column=col_name, value=temp_col)
    return dataframe
dataset = reorder_columns(dataframe=dataset, col_name='Population', position=8) # here when more elements are added change this 8
dataset.head()


# ## 2) ANALYSING DATA 

# ### DENSITY PLOT 

# In[6]:


#now lets see how Population is distributed.
#but since it is in string type so lests convert it into float dtype.
print(dataset['Population'].describe())
plt.figure(figsize=(10, 9))
sns.distplot(dataset['Population'], color='b', bins=100, hist_kws={'alpha': 0.4});


# #### OBSERVATIONS:

# Here we see that as we change the bins size the peak of graph changes,i.e if we decrease the bin size then the peak of graph decreases.
# 
# We see that the above graph is skewed, so maybe the appropriate distribution can be log or log normal distribution.
# 
# Here density represents frequency of Population

# ### COREALTION

# In[7]:


#Lets use corelation to find which features are strongly corelated with Population.
dataset[dataset.columns[0:]].corr()['Population'][:]


# VISUALIZATION ON HEAT MAPS

# In[8]:


corr = dataset.corr() # We already examined Ex-Showroom_Price correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0) | (corr <= 0)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# ## 3)APPLYING REGGRESION MODEL

# We can also use our model to predict the Population of a village with help of this dataset. So lets apply some machine learning models and compare which model gives better results.
# 
# Till now we have analysed Population. Now lets compare regression models.

# ##### IMPORTING DATASET

# In[9]:


#DECLARING FEATURES AND LABELS
x= dataset.drop(['Population'],axis=1).values
y=dataset[['Population']].values


# ##### Splitting the dataset into the Training set and Test set 

# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=0)

#here I am using random_state=0 because I will be using various models on this dataset, so using this will help
#me to get predictions on same set of training and testing data so that I can compare the results.


# ###### 3a) MULTIPLE LINEAR REGRESSION
# 

# In[11]:


#Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#Predicting the Test set results
y_pred = regressor.predict(x_test)
#Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# ###### 3b) DECISION TREE REGRESSION

# In[12]:


# Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)
#Predicting the Test set results
y_pred = regressor.predict(x_test)
#Evaluating the Model Performance
r2_score(y_test, y_pred)


# ###### 3c) RANDOM FOREST REGRESSION

# In[13]:


#Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100000, random_state = 0)
regressor.fit(x_train, y_train)
#Predicting the Test set results
y_pred = regressor.predict(x_test)
#Evaluating the Model Performance
r2_score(y_test, y_pred)


# ##### FEATURE SCALLING 

# In[14]:


#FEATURE SCALLING
Y=dataset[['Population']].values
Y = Y.reshape(len(y),1)
# splitting y into training and test dataset
y_train, y_test = train_test_split(Y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)


# ###### 3d) SUPPORT VECTOR MACHINE

# In[15]:


#Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)))
#Evaluating the Model Performance
r2_score(y_test, y_pred)


# In[ ]:




