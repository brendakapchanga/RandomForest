#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


# In[3]:


data=pd.read_csv(r'C:\Users\user\Desktop\churn\Churn_Modelling.csv')


# In[4]:


print("First five rows of the dataset:")
print(data.head())


# In[5]:


print("\nData shape:")
print(data.shape)


# In[6]:


print("\nData information:")
print(data.info())


# In[7]:


print("\nMissing values:")
print(data.isnull().sum())


# In[8]:


print("\nStatistical summary of the data:")
print(data.describe())


# In[9]:


sns.countplot(x='Exited',data=data)
plt.show()


# In[11]:


corr=data.corr()
sns.heatmap(corr,annot=True)
plt.show()


# In[12]:


geography=pd.get_dummies(data['Geography'],drop_first=True)
gender=pd.get_dummies(data['Gender'],drop_first=True)


# In[13]:


data=pd.concat([data,geography,gender],axis=1)


# In[14]:


data=data.drop(['Geography','Gender'],axis=1)


# In[24]:


X=data.drop('Exited',axis=1)
y=data['Exited']


# In[ ]:




