#!/usr/bin/env python
# coding: utf-8

# In[8]:


# 1.LOAD THE DATASET
import pandas as pd
df=pd.read_csv(r"C:\Users\Gayathri\Downloads\Churn_Modelling.csv") # import dataset
print(df)


# In[ ]:


Perform Below Visualizations.
1. Univarient Analysis
There are three ways to perform univarient analysis
i) Summary statistics


# In[11]:


# Summary statistics

import pandas as pd
df=pd.read_csv(r"C:\Users\Gayathri\Downloads\Churn_Modelling.csv")

#mean of CreditScore
M=df['CreditScore'].mean()

#median of CreditScore
Me=df['CreditScore'].median()

# standard deviation of CreditScore
std = df['CreditScore'].std()

print("mean value of CreditScore is {}".format(M))
print("median value of CreditScore is {}".format(Me))
print("Standard deviation of CreditScore is {}".format(std))


# In[ ]:


ii)FREQUENCY TABLE


# In[12]:


#Frequency table
import pandas as pd
df=pd.read_csv(r"C:\Users\Gayathri\Downloads\Churn_Modelling.csv")

#frequency table for age
ft=df['Age'].value_counts()

print("Frequency table for Age is given below")
print("{}".format(ft))


# In[ ]:


iii) CHARTS


# In[13]:


#Chart

import matplotlib.pyplot as plt
dfs = df.head() # print first five table from top
print(dfs) 

#box plot for Balance column

dfs.boxplot(column="Balance",grid=False,color="red")
plt.title('Box plot')


# In[ ]:


Histogram for Credit Score


# In[14]:


df.hist(column="CreditScore" ,grid=True, edgecolor ='black', color ='red')
plt.title('Histogram')


# In[ ]:


DENSITY CURVE


# In[15]:


import seaborn as sns  #statistical data visualization

sns.kdeplot(df['CreditScore'])
plt.title('Density Curve')


# In[ ]:


2. Bi - Variate Analysis
There are three common ways to perform bivariate analysis:
i. Scatterplots


# In[16]:


import matplotlib.pyplot as plt # library for charts

dfs1 = df.head(20)
plt.scatter(dfs1.CreditScore,dfs1.Balance)
plt.title('Scatterplots-- Banking')
plt.xlabel("CreditScore")
plt.ylabel("Balance")


# In[ ]:


ii.Correlation Coefficient


# In[17]:


df.corr()


# In[ ]:


iii. Simple Linear Regression


# In[18]:


import statsmodels.api as sm
# response variable
y = df['CreditScore']

# explanatory variable
x = df[['Balance']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())


# In[ ]:


#3. Multi - Variate Analysis


# In[67]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[55]:



i. A Matrix Scatterplot
ii. A Scatterplot with the Data Points Labelled by their Group
iii. A Profile Plot
iv. Calculating Summary Statistics for Multivariate Data
v. Means and Variances Per Group
vi. Between-groups Variance and Within-groups Variance for a Variable
vii. Between-groups Covariance and Within-groups Covariance for Two Variables
viii. Calculating Correlations for Multivariate Data
ix. Standardising Variables


# In[ ]:


#4. Perform descriptive statistics on the dataset.


# In[28]:


#load data set into ld
ld= pd.read_csv(r"C:\Users\Gayathri\Downloads\Churn_Modelling.csv")
five = ld.head() #for print first five rows


# In[24]:


# information about used data set
ld.info()


# In[29]:


# information about used data set
ld.info()


# In[ ]:


5. Handle the Missing values.


# In[30]:


ld.isnull().any()


# In[31]:


ld.isnull().sum()


# In[32]:


sns.heatmap(ld.corr(),annot=True) # heatmap -a plot of rectangular data as a color-encoded matrix


# In[ ]:


6. Find the outliers and replace the outliers


# In[33]:


#occurence of outliers
ld1= pd.read_csv(r"C:\Users\Gayathri\Downloads\Churn_Modelling.csv")
sns.boxplot(ld1.CreditScore)


# In[34]:


#Use Mean Detection and Nearest Fill Methods - Outliers

Q1= ld1.CreditScore.quantile(0.25)
Q3=ld1.CreditScore.quantile(0.75)
IQR=Q3-Q1
upper_limit =Q3 + 1.5*IQR
lower_limit =Q1 - 1.5*IQR
ld1['CreditScore'] = np.where(ld1['CreditScore']>upper_limit,30,ld1['CreditScore'])
sns.boxplot(ld1.CreditScore)


# In[ ]:


7. Check for Categorical columns and perform encoding.


# In[35]:


ld1.head(5)


# In[36]:


#label encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ld1.Gender= le.fit_transform(ld1.Gender)
ld1.head(5)


# In[37]:


#one hot encoding
ld1_main=pd.get_dummies(ld1,columns=['Geography'])
ld1_main.head()


# In[ ]:


8. Split the data into dependent and independent variables.


# In[38]:


#Splitting the Dataset into the Independent Feature Matrix
df=pd.read_csv(r"C:\Users\Gayathri\Downloads\Churn_Modelling.csv")
X = df.iloc[:, :-1].values
print(X)


# In[39]:


#Extracting the Dataset to Get the Dependent Vector
Y = df.iloc[:, -1].values
print(Y)


# In[ ]:


9. Scale the independent variables


# In[40]:


w = df.head()
q = w[['Age','Balance','EstimatedSalary']] #spliting the dataset into measureable values
q


# In[41]:


from sklearn.preprocessing import scale # library for scallling
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

x_scaled = mm.fit_transform(q)
x_scaled


# In[42]:


from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
x_ss = sc.fit_transform(q)
x_ss


# In[43]:


from sklearn.preprocessing import scale
X_scaled=pd.DataFrame(scale(q),columns=q.columns)
X_scale=X_scaled.head()
X_scale


# In[ ]:


10. Split the data into training and testing


# In[44]:


x= df[['Age','Balance','EstimatedSalary']]
x


# In[45]:


y = df['Balance']
y


# In[46]:


#scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
x_scaled1 = sc.fit_transform(x)
x_scaled1


# In[47]:


#train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled1, y, test_size = 0.3, random_state = 0)
x_train


# In[48]:


x_train.shape


# In[49]:


x_test


# In[50]:


x_test.shape


# In[51]:


y_train


# In[52]:


y_test


# In[ ]:




