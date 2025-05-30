import statsmodels.api as sm
import numpy as np
import json
import os
import pandas as pd
from ast import literal_eval
from metrics import *

df_p = pd.read_csv('prop_dev.csv')
df_a_r = pd.read_csv('resp_acc_dev.csv')
df_r_r = pd.read_csv('resp_rej_dev.csv')

prop_dum = pd.get_dummies(df_p, columns=['Model', 'Prop Belief', 'Resp Belief', 'Reasoning'])
res_acc_dum = pd.get_dummies(df_a_r, columns=['Model', 'Prop Belief', 'Resp Belief', 'Reasoning'])
res_rej_dum = pd.get_dummies(df_r_r, columns=['Model', 'Prop Belief', 'Resp Belief', 'Reasoning'])

del prop_dum['DS']
del res_acc_dum['DS']
del res_rej_dum['DS']

Y_p = df_p['DS'].astype(float)
X_p = prop_dum.astype(float)

X_p = sm.add_constant(X_p, prepend=False)

ols_p = sm.OLS(Y_p, X_p)
res_p = ols_p.fit()

print(res_p.summary())

Y_a_r = df_a_r['DS'].astype(float)
X_a_r = res_acc_dum.astype(float)

X_a_r = sm.add_constant(X_a_r, prepend=False)

ols_a_r = sm.OLS(Y_a_r, X_a_r)
res_a_r = ols_a_r.fit()

print(res_a_r.summary())

Y_r_r = df_r_r['DS'].astype(float)
X_r_r = res_rej_dum.astype(float)

X_r_r = sm.add_constant(X_r_r, prepend=False)

ols_r_r = sm.OLS(Y_r_r, X_r_r)
res_r_r = ols_r_r.fit()

print(res_r_r.summary())