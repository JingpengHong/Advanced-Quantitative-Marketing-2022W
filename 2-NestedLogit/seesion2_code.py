'''
Foundation of Advanced Quantitative Marketing
Seesion 2: Nested Logit
Data: Yogurt_Data.csv
Author: Jingpeng Hong
Instructor: Pradeep Chintagunta
'''
##################
## Housekeeping ##
##################

import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize
os.chdir("/Users/hongjingpeng/Desktop/Quant_Mkt/Advanced-Quantitative-Marketing-2022W")

####################
## Preparing Data ##
####################

### load in data from Session 1 ###
yogurt = pd.read_csv("Working_Data/yogurt_long.csv")

###################
## Data Analysis ##
###################

## Consider brands 1 to 3 in one nest and 4 in another ##

## define the nest groups ##
yogurt.loc[yogurt.brand==4, 'nest']=0
yogurt.loc[yogurt.brand!=4, 'nest']=1

def neg_log_LL(w, *args):
    
    '''
    log-likelihood function
    x: feature data, including price and feature
    y: choice data
    w: parameters
    '''
    x = args[0] ## data set
    alpha1, alpha2, alpha3, bp, bf, rho = w
    
    x['v']=alpha1*(x['brand']==1)+alpha2*(x['brand']==2)+alpha3*(x['brand']==3)+bp*x['price']+bf*x['feature']
    
    x['nest'] = 0
    x.loc[(x.brand == 1) | (x.brand == 2) | (x.brand == 3), 'nest']=1

    x.loc[x.nest==0, 'exp']=np.exp(x.v)
    x.loc[x.nest==1, 'exp']=np.exp(x.v/rho)

    exp_nest = x.groupby(['Pan I.D.', 't', 'nest'],as_index = False)['exp'].agg('sum')
    exp_nest.loc[exp_nest.nest==1, 'pnest_rho']=exp_nest['exp']**rho/3
    exp_nest.loc[exp_nest.nest==0, 'pnest_rho']=exp_nest['exp']
    exp_nest.rename(columns={"exp": "pnest"}, inplace=True)
    x=x.merge(exp_nest, how='left', on=['Pan I.D.', 't', 'nest'])

    p_sum = x.groupby(['Pan I.D.', 't'],as_index = False)['pnest_rho'].agg('sum')
    p_sum.rename(columns={"pnest_rho": "p_sum"}, inplace=True)
    x=x.merge(p_sum, how='left', on=['Pan I.D.', 't'])

    x.loc[x.nest==0, 'p']=x.exp/x.p_sum
    x.loc[x.nest==1, 'p']=x.pnest_rho*3/x.p_sum * x.exp/x.pnest
    
    # return the log-likelihood
    return -np.sum(x['choice']*np.log(x['p']))

result=minimize(neg_log_LL, [0,0,0,-1,1,0.5], args=(yogurt), method='L-BFGS-B')
coef=result.x
print(coef)
hess_inv=result.hess_inv.todense() 
se=np.diag(np.sqrt(hess_inv))

## Calculate AIC and BIC ##
print(result.fun)

## Calculate the Cross Elasticity ##
x = yogurt
alpha1, alpha2, alpha3, bp, bf, rho = coef
x['v']=alpha1*(x['brand']==1)+alpha2*(x['brand']==2)+alpha3*(x['brand']==3)+bp*x['price']+bf*x['feature']

x['nest'] = 0
x.loc[(x.brand == 1) | (x.brand == 2) | (x.brand == 3), 'nest']=1

x.loc[x.nest==0, 'exp']=np.exp(x.v)
x.loc[x.nest==1, 'exp']=np.exp(x.v/rho)

exp_nest = x.groupby(['Pan I.D.', 't', 'nest'],as_index = False)['exp'].agg('sum')
exp_nest.loc[exp_nest.nest==1, 'pnest_rho']=exp_nest['exp']**rho/3
exp_nest.loc[exp_nest.nest==0, 'pnest_rho']=exp_nest['exp']
exp_nest.rename(columns={"exp": "pnest"}, inplace=True)
x=x.merge(exp_nest, how='left', on=['Pan I.D.', 't', 'nest'])

p_sum = x.groupby(['Pan I.D.', 't'],as_index = False)['pnest_rho'].agg('sum')
p_sum.rename(columns={"pnest_rho": "p_sum"}, inplace=True)
x=x.merge(p_sum, how='left', on=['Pan I.D.', 't'])

x.loc[x.nest==0, 'p']=x.exp/x.p_sum
x.loc[x.nest==1, 'p']=x.pnest_rho*3/x.p_sum * x.exp/x.pnest
    
e = np.zeros((4,4))
for j in range(1, 5):
    for k in range(1, 5):
        y=x
        xkt = y[y.brand==k].groupby(['Pan I.D.', 't'],as_index = False)['price'].agg('mean')
        xkt.rename(columns={"price": "pricek"}, inplace=True)
        y=y.merge(xkt, how='left', on=['Pan I.D.', 't'])
        e[j-1,k-1]=-bp*np.average(y.pricek*y.p, weights=(x.brand==j))
        
print(e.round(3))

## Consider brands 1 and 4 in one nest and 2 and 3 in another ##

## define the nest groups ##

def neg_log_LL(w, *args):
    
    '''
    log-likelihood function
    x: feature data, including price and feature
    y: choice data
    w: parameters
    '''
    x = args[0] ## data set
    alpha1, alpha2, alpha3, bp, bf, rho1, rho2 = w
    
    x['v']=alpha1*(x['brand']==1)+alpha2*(x['brand']==2)+alpha3*(x['brand']==3)+bp*x['price']+bf*x['feature']
    
    x.loc[(x.brand == 1) | (x.brand == 4), 'nest'] = 1
    x.loc[(x.brand == 2) | (x.brand == 3), 'nest'] = 2

    x.loc[x.nest==1, 'exp']=np.exp(x.v/rho1)
    x.loc[x.nest==2, 'exp']=np.exp(x.v/rho2)

    exp_nest = x.groupby(['Pan I.D.', 't', 'nest'],as_index = False)['exp'].agg('sum')
    exp_nest.loc[exp_nest.nest==1, 'pnest_rho']=exp_nest['exp']**rho1/2
    exp_nest.loc[exp_nest.nest==2, 'pnest_rho']=exp_nest['exp']**rho2/2
    exp_nest.rename(columns={"exp": "pnest"}, inplace=True)
    x=x.merge(exp_nest, how='left', on=['Pan I.D.', 't', 'nest'])

    p_sum = x.groupby(['Pan I.D.', 't'],as_index = False)['pnest_rho'].agg('sum')
    p_sum.rename(columns={"pnest_rho": "p_sum"}, inplace=True)
    x=x.merge(p_sum, how='left', on=['Pan I.D.', 't'])
    
    x.loc[x.nest==1, 'p']=x.pnest_rho*2/x.p_sum * x.exp/x.pnest
    x.loc[x.nest==2, 'p']=x.pnest_rho*2/x.p_sum * x.exp/x.pnest
    
    # return the log-likelihood
    return -np.sum(x['choice']*np.log(x['p']))

result=minimize(neg_log_LL, [0,0,0,-1,1,0.5, 0.5], args=(yogurt), method='L-BFGS-B')
coef=result.x
print(coef.round(3))
hess_inv=result.hess_inv.todense() 
se=np.diag(np.sqrt(hess_inv))
print(se.round(3))

## Calculate AIC and BIC ##
print(result.fun)









