#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing the data

df = pd.read_csv("loan_default_dataset_pre-processed.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


#Splitting the data

X = df.drop(columns=["Default", "ID"], axis=1)
Y = df["Default"] 


# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# In[7]:


print(X.shape, X_train.shape, X_test.shape)


# In[8]:


print(Y.shape)


# In[9]:


#Training the Model

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)


# In[10]:


#Evaluating the Model

model.score(X_test, Y_test)


# In[11]:


Y_predicted=model.predict(X_test)


# In[12]:


cm=confusion_matrix(Y_test, Y_predicted)
cm


# In[13]:


report = classification_report(Y_test, Y_predicted, digits=3)
print(report)


# In[ ]:




