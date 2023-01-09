#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Importing the data

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


#Training the Model

clf_entropy=DecisionTreeClassifier(criterion='entropy',random_state=100, max_depth=3, min_samples_leaf=5)


# In[8]:


clf_entropy.fit(X_train, Y_train)


# In[9]:


# Evaluating the Model

clf_entropy.score(X_test, Y_test)


# In[10]:


Y_predicted=clf_entropy.predict(X_test)


# In[11]:


cm=confusion_matrix(Y_test, Y_predicted)
cm


# In[12]:


report = classification_report(Y_test, Y_predicted, digits=3)
print(report)


# In[ ]:




