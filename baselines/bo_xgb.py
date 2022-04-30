# -*- coding: utf-8 -*-
"""bo_xgb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18kFihac0U1JML8kEoxQDV3f1TUBOqVa0
"""

!pip install xgboost
!pip install bayesian-optimization

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

## Hyperparameter optimization using Bayesian Optimization

import xgboost as xgb
from scipy.stats import loguniform
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/drive')

#%cd '/content/drive/My Drive/545'

df = pd.read_csv(r'Churn_Modelling.csv')
#df.head()

#df.shape

#Getting the independent features
X = df.iloc[:,3:-1]
#Getting the dependent feature
y = df.iloc[:,-1]

#Dummy variable for 'Geography' column
geography = pd.get_dummies(X['Geography'], drop_first = True)
#Dummy variable for 'Gender' column
gender = pd.get_dummies(X['Gender'], drop_first = True)

#Dropping the original 'Geography' and 'Gender' columns
X = X.drop(['Geography','Gender'], axis = 1)

#Adding the dummy columns to the dataset
X = pd.concat([X,geography,gender], axis = 1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

global_parms = {"booster"         : "gbtree", 
                "missing"         : None,
                "n_estimators"    : 100, 
                "n_jobs"          : 1, 
                "objective"       : 'binary:logistic', 
                "random_state"    : 545, 
                "scale_pos_weight": 1, 
                "verbosity"       : 1,
                "eval_metric"     : 'logloss'
}

params_space={
 "base_score"       : (0.1,0.9),
 "colsample_bylevel": (0.1,1),
 "colsample_bynode" : (0.1,1),
 "colsample_bytree" : (0.1,1),
 "learning_rate"    : (0.1, 1),
 "max_delta_step"   : (0, 10), # integer
 "max_depth"        : (1, 20), # integer
 "min_child_weight" : (1, 20),
 "gamma"            : (0.1,10),
 "subsample"        : (0.1,1),
 "reg_lambda"       : (1, 9),
 "reg_alpha"        : (1, 9),
}

def bo_tune_xgb(base_score, colsample_bylevel,colsample_bynode,colsample_bytree,learning_rate,max_delta_step,max_depth,min_child_weight,gamma,subsample,reg_lambda,reg_alpha):
      xgb_params = {
              'base_score': base_score,
              'colsample_bylevel':colsample_bylevel,
              'colsample_bytree':colsample_bytree,
              'colsample_bynode':colsample_bynode,
              'learning_rate': learning_rate,
              'subsample': subsample,
              'gamma': gamma,
              'min_child_weight': min_child_weight,
              'max_depth': int(max_depth),
              'max_delta_step':int(max_delta_step),
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              }   
      param_all = global_parms.copy()
      param_all.update(xgb_params)
      cv_result = xgb.cv(param_all, dtrain, num_boost_round= 100, nfold=5)
      return -1.0 * cv_result['test-logloss-mean'].iloc[-1]

from sklearn.metrics import log_loss, accuracy_score
def run_xgb(params, dtrain, dtest):
  param_all = global_parms.copy()
  param_all.update(params)
  watchlist = [(dtest,'eval'), (dtrain,'train')]
  evals_result = {}
  xgb_model = xgb.train(param_all, 
                      dtrain, 
                      num_boost_round=100,
                      evals=watchlist,
                      evals_result=evals_result,
                      verbose_eval = False)
  y_pred = xgb_model.predict(dtest)

  return evals_result['eval']['logloss'][-1], accuracy_score(y_test, np.round(y_pred))

rand_run_list = [0, 100, 200, 300, 400]
#rand_run_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
for count, random_state in enumerate(rand_run_list):
    xgb_bo = BayesianOptimization(bo_tune_xgb, params_space, random_state=random_state)
    xgb_bo.maximize(init_points=50, n_iter=200,acq='ucb')
    best_params = xgb_bo.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['max_delta_step'] = int(best_params['max_delta_step'])

    bo_val_loss = []
    for i,res in enumerate(xgb_bo.res):
        bo_val_loss.append(-1.0 * xgb_bo.res[i]['target'])    

    loss = np.array(bo_val_loss)
    loss_minimum_accumulate = np.minimum.accumulate(loss)
    df = pd.DataFrame(loss_minimum_accumulate).transpose()
    df.to_csv("BO_tune_xgb_new2.csv", mode = 'a', index = False, header = False)

    eval_result = run_xgb(best_params, dtrain, dtest)
    df_test = pd.DataFrame(eval_result).transpose()
    df_test.to_csv("BO_test_xgb_new2.csv", mode = 'a', index = False, header = False)

