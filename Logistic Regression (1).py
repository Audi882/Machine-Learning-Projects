#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
# - Goal is to create a classification model which can predict all positive classes correctly.(Recall Should be high.)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 250
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/aishwaryamate/Datasets/main/Client.csv', index_col=0)
df


# In[3]:


df.drop(columns=['CASENUM'], inplace=True)


# In[4]:


df


# In[5]:


df.describe()


# In[6]:


#Check missing values


# In[7]:


df.isna().sum()


# In[8]:


#Missing value imputation


# In[9]:


from sklearn.impute import SimpleImputer


# In[10]:


si = SimpleImputer(strategy='most_frequent')


# In[11]:


df.head()


# In[12]:


df.iloc[:,1:4]


# In[13]:


df.iloc[:,1:4] = si.fit_transform(df.iloc[:,1:4])


# In[14]:


df.isna().sum()


# In[15]:


df['CLMAGE'].fillna(df['CLMAGE'].median(), inplace=True)


# In[16]:


df.isna().sum()


# In[17]:


#Define x and y


# In[18]:


df


# In[19]:


x = df.iloc[:,1:]
y = df['ATTORNEY']
y


# # Model Building

# In[20]:


#Split the data.


# In[21]:


df.head(10)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[24]:


xtrain


# In[25]:


ytrain


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


lg = LogisticRegression()


# In[28]:


lg.fit(xtrain,ytrain)


# In[29]:


xtrain.head()


# In[30]:


lg.coef_


# In[31]:


xtest.head()


# In[32]:


(-0.36427295)*1.0+(-0.335242)*1.0+( 0.70473014)*0.0+(-0.00883013)*5.0+(0.41189508)*5.163


# In[33]:


1/(1+np.exp(-1.3829486980400003))


# In[34]:


(-0.36427295)*1.0+(-0.335242)*1.0+( 0.70473014)*0.0+(-0.00883013)*41.0+(0.41189508)*0.671


# In[35]:


1/(1+np.exp(0.7851686813200001))


# In[36]:


xtest.head(3)


# In[37]:


len(xtest)


# In[39]:


ypred = lg.predict(xtest)


# In[40]:


print(ytest[:25].values)
print(ypred[:25])


# In[41]:


from sklearn.metrics import confusion_matrix, classification_report


# In[42]:


confusion_matrix(ytest,ypred) #Sequence -> tn, fp, fn, tp


# In[43]:


sns.heatmap(confusion_matrix(ytest,ypred),annot=True, fmt='g')


# In[44]:


print(classification_report(ytest,ypred))


# # Threshold Selection

# In[46]:


xtest.head(3)


# In[48]:


ypred[:5]


# In[50]:


proba = lg.predict_proba(xtest)[:,1]


# In[51]:


from sklearn.metrics import roc_auc_score,roc_curve, accuracy_score


# In[78]:


auc = roc_auc_score(ytest,proba)
auc


# In[54]:


fpr,tpr,threshold = roc_curve(ytest,proba)


# In[55]:


fpr


# In[56]:


tpr


# In[57]:


threshold


# In[77]:


plt.plot(fpr,tpr, label = 'AUC: %0.2f'%auc)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()


# In[59]:


proba[:5]


# In[60]:


accuracy = []

for i in threshold:
    ypred = np.where(proba>i,1,0)
    accuracy.append(accuracy_score(ytest,ypred))


# In[61]:


accuracy


# In[62]:


threshold_selection = pd.DataFrame({
    'Threshold' : threshold,
    'Accuracy' : accuracy
})


# In[63]:


threshold_selection


# In[64]:


threshold_selection.sort_values(by = 'Accuracy',ascending=False)


# In[65]:


from sklearn.preprocessing import binarize


# In[66]:


proba


# In[72]:


new_pred = binarize([proba],threshold=0.435684)[0]


# In[73]:


new_pred


# In[74]:


print(classification_report(ytest,new_pred))


# In[ ]:




