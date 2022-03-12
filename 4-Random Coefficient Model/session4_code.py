'''
Foundation of Advanced Quantitative Marketing
Seesion 4: 
    1) Random Coefficients Models
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
os.chdir("/Users/hongjingpeng/Desktop/Quant_Mkt/Advanced-Quantitative-Marketing-2022W/")

####################
## Preparing Data ##
####################

## load original data ##
df = pd.read_csv("Working_Data/yogurt_working.csv")

##############################
## Random Coefficient Model ##
##############################

nK = 5 # number of attributes
nJ = 4 # number of brands
nN = int(len(df)/nJ) # number of purchase occasions
 
def RC(w, *args):
    '''
    Return the -log likelihood of the Random Coefficients Models
    Parameters:
        w - the list of parameters
            first nK are means, and the rest are covariance matrix
        args[0] - original data
                  each row represent each choice at each purchase occasion
    Pre-specified coefficients:
        nJ - number of brands (choices)
        nK - number of attributes
        D - number of draws
        Q_std - standard normal draw
    '''
    
    data = args[0]
    
    ## reshape w ##
    w_mat = np.array(w).reshape(nK + 1, nK)
    theta = w_mat[0, :] # means of the MVN of coefficients
    gamma = w_mat[1:, :] # Cholesky decomposition of the covariance matrix
    
    # add mean and variance to standard normal draws ##
    Q = np.tile(theta, (D, 1)) + Q_std @ gamma 
    
    ## compute the likelihhod ##

    attribute = data[['brand_1', 'brand_2', 'brand_3', 'Feature', 'Price']]
    u = attribute @ Q.T # utility
    expu = np.exp(u)
    t = data['t']
    exp = pd.concat([t, expu], axis = 1)
    exp_sum = exp.groupby('t')[list(range(D))].agg('sum')
    exp_sum_dup = np.tile(exp_sum, nJ).reshape(nJ*len(exp_sum), D)
    p = expu / exp_sum_dup # purchase probability for each brand
    data = pd.concat([data, p], axis = 1)
    
    data.loc[(data['choice'] == 0), p.columns] = 1
    L = data.groupby('Pan I.D.', as_index = False)[p.columns].prod()

    Li = L.iloc[:,1:].sum(axis = 1) / D # household level likelihood
    
    return -sum(np.log(Li))

################
## Estimation ##
################

## create initial values ##
theta0 = [0] * nK
gamma0 = np.triu(np.ones(nK)).reshape(-1).tolist()
int_val = theta0 + gamma0

##############
## 30 Draws ##
##############

D = 30
# standard normal draw
Q_std = np.random.normal(size = (D, nK)) 

# estimation
result30 = sp.optimize.minimize(RC, int_val, args=(df), method='L-BFGS-B')

coef30 = result30.x
coef30 = np.array(coef30).reshape(nK + 1, nK)
theta30 = coef30[0, :] 
gamma30 = coef30[1:, :]
print(theta30)
print(abs(np.diag(gamma30)))

##############
## 50 Draws ##
##############

D = 50
# standard normal draw
Q_std = np.random.normal(size = (D, nK)) 

# estimation
result50 = sp.optimize.minimize(RC, int_val, args=(df), method='L-BFGS-B')

coef50 = result50.x
coef50 = np.array(coef50).reshape(nK + 1, nK)
theta50 = coef50[0, :] 
gamma50 = coef50[1:, :]
print(theta50)
print(abs(np.diag(gamma50)))


###############
## 100 Draws ##
###############

D = 100
# standard normal draw
Q_std = np.random.normal(size = (D, nK)) 

# estimation
result100 = sp.optimize.minimize(RC, int_val, args=(df), method='L-BFGS-B')

coef100 = result100.x
coef100 = np.array(coef100).reshape(nK + 1, nK)
theta100 = coef100[0, :] 
gamma100 = coef100[1:, :]
print(theta100)
print(abs(np.diag(gamma100)))

