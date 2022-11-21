# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:41:24 2022

@author: Gopinath
"""


#######question 1



import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
# Loading the data
df = pd.read_csv("delivery_time.csv")
df.shape

# scatter plot
df.plot.scatter(x='Sorting Time', y='Delivery Time')
df=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
df
df.info()
df.describe

## correlation analysis
df.corr()
#splitting of data
train = df.head(15) # training data
test = df.tail(5) # test Data
## data visualization
sns.regplot(x=df['sorting_time'],y=df['delivery_time'])
## model fitting
model=smf.ols("delivery_time~sorting_time",data=df).fit()
# Finding Coefficient parameters
model.params
# Finding tvalues and pvalues
model.tvalues , model.pvalues
# Finding Rsquared Values
model.rsquared , model.rsquared_adj
## model prediction
new_data=df.iloc[:,1]
new_data
data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred
pred_data=model.predict(data_pred)
pred_data

#splitting of data
train = df.head(16)
train # training data
test = df.tail(5) # test Data
test
#Transformation model
#1.linear
import statsmodels.formula.api as smf
df1=df.copy()
linear=smf.ols('delivery_time~sorting_time',data=df1).fit()
pred_linear=pd.Series(linear.predict(pd.DataFrame(test['sorting_time'])))
rmse_linear=np.sqrt(np.mean((np.array(test['sorting_time']-np.array(pred_linear))**2)))
rmse_linear
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df1)
#histogram
pyplot.subplot(212)
pyplot.hist(df1)
pyplot.show()
#2.exponential
df2=df.copy()
df2['log_delivery_time'] = np.log(df2['delivery_time'])
df2
exp=smf.ols('log_delivery_time~sorting_time',data=df2).fit()
pred_exp=pd.Series(exp.predict(pd.DataFrame(test['sorting_time'])))
rmse_exp=np.sqrt(np.mean((np.array(test['sorting_time'])-np.array(np.exp(pred_exp)))**2))
rmse_exp
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df2)
#histogram
pyplot.subplot(212)
pyplot.hist(df2)
pyplot.show()

#3.square root transform
from numpy import sqrt
from pandas import DataFrame
df3=df.copy()
df3
df3['sqrt_delivery_time'] = np.sqrt(df3['delivery_time'])
df3
exp=smf.ols('sqrt_delivery_time~sorting_time',data=df3).fit()
pred_sqrt=pd.Series(exp.predict(pd.DataFrame(test['sorting_time'])))
rmse_sqrt=np.sqrt(np.mean((np.array(test['sorting_time'])-np.array(np.exp(pred_sqrt)))**2))
rmse_sqrt
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df3)
#histogram
pyplot.subplot(212)
pyplot.hist(df3)
pyplot.show()

#4.cube root transform
from numpy import cbrt
from pandas import DataFrame
df4=df.copy()
df4
df4['cbrt_delivery_time'] = np.cbrt(df4['delivery_time'])
df4
exp=smf.ols('cbrt_delivery_time~sorting_time',data=df4).fit()
pred_cbrt=pd.Series(exp.predict(pd.DataFrame(test['sorting_time'])))
rmse_cbrt=np.sqrt(np.mean((np.array(test['sorting_time'])-np.array(np.exp(pred_sqrt)))**2))
rmse_cbrt
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df4)
#histogram
pyplot.subplot(212)
pyplot.hist(df4)
pyplot.show()

x=pd.Series([rmse_linear,rmse_exp,rmse_sqrt,rmse_cbrt],['linear', 'exponential', 'square root','cube root'])
out=pd.DataFrame(x)
print(out)
#Based on the results of rmse values(rmse estimates the deviation of the actual y-values from the regression line)
#for the above transformation models,exponential followed by linear is the best prediction model for the given dataset
















######question 2


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
# Loading the data
df = pd.read_csv("Salary_Data.csv")
df.shape
df.info()
df.describe
# Data visualization
df.plot.scatter(x='YearsExperience', y='Salary')
df
# correlation analysis
df.corr()
# model fitting
model=smf.ols("Salary~YearsExperience",data=df).fit()
# Finding Coefficient parameters
model.params
# Finding tvalues and pvalues
model.tvalues , model.pvalues
# Finding Rsquared Values
model.rsquared , model.rsquared_adj

# model prediction
new_data=df.iloc[:,0]
new_data
data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred
model.predict(data_pred)
#splitting of data
train = df.head(25)
train # training data
test = df.tail(5) # test Data
test
#Transformation model
#1.linear
import statsmodels.formula.api as smf
df1=df.copy()
linear=smf.ols('Salary~YearsExperience',data=df1).fit()
pred_linear=pd.Series(linear.predict(pd.DataFrame(test['YearsExperience'])))
rmse_linear=np.sqrt(np.mean((np.array(test['YearsExperience']-np.array(pred_linear))**2)))
rmse_linear
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df1)
#histogram
pyplot.subplot(212)
pyplot.hist(df1)
pyplot.show()
#2.exponential
df2=df.copy()
df2['log_Salary'] = np.log(df2['Salary'])
df2
exp=smf.ols('log_Salary~YearsExperience',data=df2).fit()
pred_exp=pd.Series(exp.predict(pd.DataFrame(test['YearsExperience'])))
rmse_exp=np.sqrt(np.mean((np.array(test['YearsExperience'])-np.array(np.exp(pred_exp)))**2))
rmse_exp
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df2)
#histogram
pyplot.subplot(212)
pyplot.hist(df2)
pyplot.show()

#3.square root transform
from numpy import sqrt
from pandas import DataFrame
df3=df.copy()
df3
df3['sqrt_Salary'] = np.sqrt(df3['Salary'])
df3
exp=smf.ols('sqrt_Salary~YearsExperience',data=df3).fit()
pred_sqrt=pd.Series(exp.predict(pd.DataFrame(test['YearsExperience'])))
rmse_sqrt=np.sqrt(np.mean((np.array(test['YearsExperience'])-np.array(np.exp(pred_sqrt)))**2))
rmse_sqrt
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df3)
#histogram
pyplot.subplot(212)
pyplot.hist(df3)
pyplot.show()

#4.cube root transform
from numpy import cbrt
from pandas import DataFrame
df4=df.copy()
df4
df4['cbrt_Salary'] = np.cbrt(df4['Salary'])
df4
exp=smf.ols('cbrt_Salary~YearsExperience',data=df4).fit()
pred_cbrt=pd.Series(exp.predict(pd.DataFrame(test['YearsExperience'])))
rmse_cbrt=np.sqrt(np.mean((np.array(test['YearsExperience'])-np.array(np.exp(pred_sqrt)))**2))
rmse_cbrt
#lineplot
import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.plot(df4)
#histogram
pyplot.subplot(212)
pyplot.hist(df4)
pyplot.show()

x=pd.Series([rmse_linear,rmse_exp,rmse_sqrt,rmse_cbrt],['linear', 'exponential', 'square root','cube root'])
out=pd.DataFrame(x)
print(out)
#Based on the results of rmse values(rmse estimates the deviation of the actual y-values from the regression line)
# RMSE can range from 0 to ∞.Lower values are better.
#for the above transformation models,exponential followed by linear is the best prediction model for the given dataset














