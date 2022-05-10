'''
Foundation of Advanced Quantitative Marketing
Seesion 6: 
    1) BLP: Aggregate Level Data
Data: Coffee_Data.csv
Author: Jingpeng Hong
Instructor: Pradeep Chintagunta
'''

##################
## Housekeeping ##
##################

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.iolib.summary2 import summary_col

os.chdir("/Users/hongjingpeng/Desktop/Quant_Mkt/Advanced-Quantitative-Marketing-2022W/")

####################
## Preparing Data ##
####################

df = pd.read_csv("Data/Coffee_Data.csv")

## reshape data to each brand each week in each column ##

df2 = df
colname = df2.columns.tolist()
df2 = pd.wide_to_long(df2, stubnames=["Price", "Sales", "feat", "disp", "fd"], i=colname[-6:], j="brand", sep="")
df2 = df2.reset_index()

## brand dummy ##
brand_dummy = pd.get_dummies(df2['brand'], prefix='brand')
df2 = pd.concat([df2, brand_dummy], axis=1)

## time dummy ##
nJ = len(set(df2['brand'])) # number of brands
nN = df2.shape[0] # number of observations
nT = int(nN / nJ) # number of periods
df2['t'] = np.repeat(list(range(nT)), nJ, axis=0).flat

df2.to_csv('Working_Data/coffee_working.csv', index=False)  

##########################
## Logistic Regression  ##
## OLS and 2SLS         ##
##########################

df3 = df2

## calculate the share of non-purchase option ##
Q = 500000 # arbitary market size
df3['share'] = df3['Sales'] / Q
df3['s0'] = 1- df3.groupby(by=['t']).transform('sum')['share']

## variable prep ##

#########
## OLS ##
#########

y = np.log(df3['share']/df3['s0'])
x = df3[['Price', 'feat', 'disp', 'fd', 'brand_1', 'brand_2', 'brand_3']]
x = sm.add_constant(x)
model_ols = sm.OLS(y, x).fit()

##########
## 2SLS ##
##########

exog = df3[['feat', 'disp', 'fd', 'brand_1', 'brand_2', 'brand_3']]
exog = sm.add_constant(exog)
endog = df3['Price']
iv = df3[['spot1', 'spot2', 'spot3', 'spot4', 'spot5', 'spot6']]

iv_stage1 = sm.OLS(endog, pd.concat([exog, iv], axis = 1)).fit()
price_hat = pd.DataFrame(iv_stage1.predict())
iv_stage2 = sm.OLS(y, pd.concat([exog, price_hat], axis = 1)).fit()

result = summary_col([model_ols, iv_stage2],stars=True,float_format='%0.4f')
print(result.as_latex())
























