#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Importing all the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')


# In[70]:


# Importing the data

df = pd.read_csv("loan_default_dataset_pre-processed.csv")


# In[71]:


df.shape


# In[72]:


df.head()


# In[73]:


#Splitting the data

X = df.drop(columns=["Default", "ID"], axis=1)
Y = df["Default"]


# In[74]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


# In[75]:


clf=GaussianNB()


# In[76]:


clf.fit(X_train, Y_train)


# In[77]:


clf.score(X_train,Y_train)


# In[78]:


Y_pred=clf.predict(X_test)


# In[80]:


accuracy=accuracy_score(Y_test,Y_pred)
print("Accuracy:", accuracy)


# In[81]:


cm=confusion_matrix(Y_test, Y_pred)
cm


# In[82]:


report = classification_report(Y_test, Y_pred, digits=3)
print(report)


# In[ ]:




