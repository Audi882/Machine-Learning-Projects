#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 250
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#read the data
df = pd.read_csv('https://raw.githubusercontent.com/aishwaryamate/Datasets/main/Cars.csv')
df


# In[4]:


df.describe()


# In[5]:


df.isna().sum()


# In[4]:


#Data visualization


# In[5]:


sns.pairplot(df)


# In[6]:


#Correlation


# In[4]:


sns.heatmap(df.corr(),annot=True)


# # Model Building

# In[5]:


import statsmodels.formula.api as smf


# In[9]:


model = smf.ols('MPG~HP+VOL+SP+WT',data=df).fit()


# In[10]:


model.pvalues


# # Simple Linear Regression

# In[11]:


#Wt
wt = smf.ols('MPG~WT+HP+SP',data = df).fit()
wt.pvalues


# In[12]:


#vol
vol = smf.ols('MPG~VOL+SP+HP',data = df).fit()
vol.pvalues


# # Calculate VIF

# In[6]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[7]:


df.head()


# In[8]:


data = df.drop(columns=['MPG'])
data


# In[9]:


data.values


# In[10]:


vif = pd.DataFrame()
vif


# In[11]:


vif['Features'] = data.columns
vif


# In[12]:


range(len(data.columns))


# In[13]:


vif['VIF'] = [variance_inflation_factor(data.values,i) for i in range(len(data.columns))]


# In[21]:


vif


# # Subset Selection
# 
# AIC

# In[22]:


#wt
wt = smf.ols('MPG~WT+HP+SP',data = df).fit()
wt.rsquared, wt.aic


# In[23]:


#vol
vol = smf.ols('MPG~VOL+SP+HP',data = df).fit()
vol.rsquared, vol.aic


# In[24]:


model.summary()


# # MODEL VALIDATION TECHNIQUES

# # 1.Residual Analysis
# 
# 
# **As per the subset selection, it is clear that VOL column is more important than WT column as it's R2 value is more and AIC value is less,So we will select VOL columns and will drop WT column.**
# 
# **NORMALITY TEST**

# In[15]:


model = smf.ols('MPG~HP+VOL+SP', data=df).fit()
model.rsquared


# In[16]:


#Statsmodel
import statsmodels.api as sm


# In[17]:


df.head()


# In[18]:


model.resid


# In[19]:


#Q-Q plot
sm.qqplot(model.resid, line = 'q');


# In[27]:


#get index for higher residuals


# In[46]:


np.argmax(model.resid), np.max(model.resid)


# # 2. Residual plot of Homoscedasticity
# 
# - Homoscedasticity can be checked by plotting a scatter plot between fitted values and residuals.

# In[48]:


model.fittedvalues, model.resid


# In[49]:


plt.scatter(model.fittedvalues, model.resid)


# # 3.Residual VS Regressor

# In[51]:


#Vol
sm.graphics.plot_regress_exog(model,'VOL');


# In[53]:


sm.graphics.plot_fit(model,'VOL');


# In[54]:


#Sp
sm.graphics.plot_fit(model,'SP');


# In[55]:


#Hp
sm.graphics.plot_fit(model,'HP');


# # MODEL DELETION TECHNIQUES

# # Cook's Distance
#    - **Detecting influencers and outliers**

# In[56]:


model


# In[59]:


#Find the influence data
inf = model.get_influence()

#Calculate the cooks distance
c , p = inf.cooks_distance


# In[60]:


c


# In[61]:


# Cook's distance plot
plt.stem(c)


# In[63]:


np.argmax(c), np.max(c)


# In[65]:


df.iloc[[76]]


# In[66]:


df.head()


# In[20]:


#Influence Plot
from statsmodels.graphics.regressionplots import influence_plot


# In[69]:


influence_plot(model);


# In[73]:


#Calculate cutoff
k = len(df.columns)
n = len(df)

lv = 3*(k+1)/n
lv


# In[76]:


influence_plot(model)
plt.axvline(lv, linestyle = '--', color = 'red')


# # Improving the model

# In[77]:


df.drop(index=76,inplace=True)


# In[78]:


df


# In[35]:


#reset the index


# In[79]:


df.reset_index(inplace=True)


# In[80]:


df


# In[81]:


df.drop(columns=['index'], inplace=True)


# In[82]:


df


# In[83]:


final_model = smf.ols('MPG~HP+SP+VOL', data = df).fit()


# In[84]:


final_model.rsquared


# In[36]:


# Cook's distance plot


# In[ ]:





# In[37]:


#Final Model


# **Since the value is <1 , we can stop the diagnostic process and finalize the model**

# # Predicting for new records

# In[86]:


df.head()


# In[87]:


test = pd.DataFrame({
    'HP' : [56,53.62,95],
    'VOL' : [92.6,85.63,75],
    'SP' : [110,112,150]
})


# In[89]:


test


# In[88]:


final_model.predict(test)

