#!/usr/bin/env python
# coding: utf-8

# # Salary Data Analysis by SVM

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC


# ## Reading the File

# In[2]:


train = pd.read_csv('D:/Data Science/03-04June/12. Support Vector Machines/SalaryData_Train.csv')


# In[3]:


train.head(5)


# In[4]:


test = pd.read_csv('D:/Data Science/03-04June/12. Support Vector Machines/SalaryData_Test.csv')


# In[5]:


test.head(5)


# ## Basic Info Data

# In[6]:


train.info()


# In[7]:


test.info()


# ## Checking the Duplicates

# In[142]:


train.duplicated().sum()


# In[143]:


test.duplicated().sum()


# ## Removing the duplicates

# In[144]:


train.drop_duplicates(inplace = True)


# In[145]:


test.drop_duplicates(inplace = True)


# In[146]:


train.shape ### 3258 duplicates removed


# In[147]:


test.shape ### 930 duplicates removed


# ## Checking duplicates

# In[16]:


train.isnull().sum()


# In[17]:


test.isnull().sum()


# ## Unique Value Counts

# In[148]:


print(train['workclass'].value_counts())
print(train['education'].value_counts())
print(train['maritalstatus'].value_counts())
print(train['occupation'].value_counts())
print(train['relationship'].value_counts())
print(train['race'].value_counts())
print(train['sex'].value_counts())
print(train['native'].value_counts())
print(train['Salary'].value_counts())


# In[149]:


print(test['workclass'].value_counts())
print(test['education'].value_counts())
print(test['maritalstatus'].value_counts())
print(test['occupation'].value_counts())
print(test['relationship'].value_counts())
print(test['race'].value_counts())
print(test['sex'].value_counts())
print(test['native'].value_counts())
print(test['Salary'].value_counts())


# # Exploratory Data Analysis

# In[8]:


sns.countplot(x='Salary',data=train, hue='workclass')


# ## Salary of Private employees is higher then other work class in train data

# In[9]:


sns.countplot(x='Salary',data=test, hue='workclass')


# ## Salary of Private employees is higher then other work class in test data

# In[10]:


pd.crosstab(train['occupation'],train['Salary']).plot(kind='bar',title='Occupation wise Salary in train data')


# In[153]:


pd.crosstab(test['occupation'],test['Salary']).plot(kind='bar',title='Occupation wise Salary in test data')


# In[154]:


pd.crosstab(train['education'],train['Salary']).plot(kind='barh',title='Education wise Salary in train data')


# In[155]:


pd.crosstab(test['education'],test['Salary']).plot(kind='barh',title='Education wise Salary in test data')


# In[157]:


sns.lineplot(x='age',y='hoursperweek',data=train)


# In[158]:


sns.lineplot(x='age',y='hoursperweek',data=test)


# ## At the age 40 - 60 the hrs per week is greater then 40

# In[159]:


pd.crosstab(train['race'],train['sex']).plot(kind='barh',title='Race wise Gender in train data')


# In[160]:


pd.crosstab(test['race'],test['sex']).plot(kind='barh',title='Race wise Gender in test data')


# In[161]:


sns.countplot(x='sex',data=train)


# ## Male and Female distribution in train data

# In[162]:


sns.countplot(x='sex',data=test)


# In[ ]:


## Male and Female distribution in test data


# In[163]:


train['Salary'].value_counts().plot(kind='pie',title='Salary in Train Data',autopct = "%0.0f%%",colors=['blue','orange'])


# In[164]:


test['Salary'].value_counts().plot(kind='pie',title='Salary in Test Data',autopct = "%0.0f%%",colors=['lightgrey','grey'])


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


le = LabelEncoder()


# In[13]:


train['workclass'] = le.fit_transform(train['workclass'])
train['education'] = le.fit_transform(train['education'])
train['maritalstatus'] = le.fit_transform(train['maritalstatus'])
train['occupation'] = le.fit_transform(train['occupation'])
train['relationship'] = le.fit_transform(train['relationship'])
train['sex'] = le.fit_transform(train['sex'])
train['race'] = le.fit_transform(train['race'])
train['native'] = le.fit_transform(train['native'])


# In[14]:


train.head(5)


# In[15]:


test['workclass'] = le.fit_transform(test['workclass'])
test['education'] = le.fit_transform(test['education'])
test['maritalstatus'] = le.fit_transform(test['maritalstatus'])
test['occupation'] = le.fit_transform(test['occupation'])
test['relationship'] = le.fit_transform(test['relationship'])
test['sex'] = le.fit_transform(test['sex'])
test['race'] = le.fit_transform(test['race'])
test['native'] = le.fit_transform(test['native'])


# In[16]:


test.head(5)


# ## Applying SVM

# In[52]:


x=train.iloc[:,0:13]
x.shape


# In[51]:


y=train.iloc[:,13]
y.shape


# In[53]:


model_svm = SVC()


# In[54]:


clf = model_svm.fit(x,y)
pred = clf.predict(x)


# In[67]:


train['Pred_Salary']=pred
train


# In[69]:


accuracy_score(train['Salary'],train['Pred_Salary'])*100


# ## After applying SVM model is predicting 79% accuracy on train data

# ## Lets check it with Test data

# In[70]:


x=test.iloc[:,0:13]
x.shape


# In[71]:


y=test.iloc[:,13]
y.shape


# In[72]:


model_svm1 = SVC()


# In[73]:


clf1 = model_svm1.fit(x,y)
pred1 = clf1.predict(x)


# In[74]:


accuracy_score(y,pred1)*100


# ## After applying SVM model is predicting 79% accuracy on test data
