# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:17:49 2022

@author: zixiaoxx
"""

# In[19]:

from collections import Counter
#import hyperopt
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import numpy as np
import scipy as sp
import pandas as pd
import warnings

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from functools import partial
import time
# In[89]:

#%%
import os
warnings.filterwarnings('ignore')


X_train_non_null = pd.read_csv("X_train.csv")
#X_test_non_null = pd.read_csv("X_test.csv")

y = X_train_non_null['AdoptionSpeed']
X = X_train_non_null.drop(['AdoptionSpeed'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
dtrain = xgb.DMatrix(X_train, label = y_train)

#%%
'''
def QWK(predt: np.ndarray, dtrain: xgb.DMatrix):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = dtrain.get_label()
    rater_b = predt
    optR = OptimizedRounder()
    optR.fit(rater_b, rater_a)
    coefficients = optR.coefficients()
    rater_b = optR.predict(rater_b, coefficients)
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return "QWK", -(1.0 - numerator / denominator)

'''
# In[104]:

space = {
          'eta':(0.01,0.2),
          # 'gamma':(0,25),
          # 'max_depth':(2,18),
          # 'min_child_weight':(0,20),
          'max_delta_step':(0,10),
          'subsample':(0.5,1),
          # 'colsample_bytree':(0.5,1),
          # 'reg_alpha' :(40,180),
          'reg_lambda' :(0,1),

          }

xgb_default_params = {
    'eval_metric': 'mlogloss',
    'seed': 1337,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': True,
    'num_class': 5,
    'n_estimator' : 180,
}
# In[105]:
'''
def bo_tune_xgb(eta, gamma, max_depth, min_child_weight, max_delta_step,subsample,colsample_bytree,reg_alpha,reg_lambda ):
    xgb_param = {
              'eval_metric': 'mlogloss',
              'max_depth': int(max_depth),
              'subsample': subsample,
              'eta': eta,
              'gamma': int(gamma),
              'min_child_weight':int(min_child_weight),
              'max_delta_step':int(max_delta_step),
              'colsample_bytree':colsample_bytree,
              'reg_alpha' :int(reg_alpha),
              'reg_lambda' :reg_lambda,
    'seed': 1337,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': True,
    'verbosity': 0,
    'n_estimator' : 180,
    'verbosity' : 0,
    'num_class': 5
              }   
    cv_result = xgb.cv(xgb_param, dtrain, num_boost_round= 100, nfold=4)
    # return -1.0 * cv_result['test-merror-mean'].iloc[-1]
    return -1.0 * cv_result['test-mlogloss-mean'].iloc[-1]
'''
#%%
def bo_tune_xgb(eta, max_delta_step,subsample,reg_lambda ):
    xgb_param = {
              'eval_metric': 'mlogloss',
              'eta': eta,
              'max_delta_step':int(max_delta_step),
              'reg_lambda' :reg_lambda,
    'seed': 1337,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': True,
    'verbosity': 0,
    'n_estimator' : 180,
    'verbosity' : 0,
    'num_class': 5
              }   
    cv_result = xgb.cv(xgb_param, dtrain, num_boost_round= 70, nfold=4)
    # return -1.0 * cv_result['test-merror-mean'].iloc[-1]
    return -1.0 * cv_result['test-mlogloss-mean'].iloc[-1]
#%%
start = time.time()
xgb_bo = BayesianOptimization(bo_tune_xgb, space)
xgb_bo.maximize(init_points = 10, n_iter = 500, acq = 'ei')

# In[107]:
best_param = xgb_bo.max['params']
print(best_param)
best_param['max_depth'] = int(best_param['max_depth'])

end = time.time()
print("The execution time of BO is :", end-start)
#%%
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']

#%%


n_splits = 10
num_rounds = 60000
early_stop = 500


best_params ={'colsample_bytree': 0.9767822931505248, 'eta': 0.168557552753834, 'gamma': 0.794021094267594, 'max_delta_step': 0.17208321684861416, 'max_depth': 12.149100826622568, 'min_child_weight': 9.751329446686068, 'reg_alpha': 41.597153298924816, 'reg_lambda': 0.849251054732031, 'subsample': 0.9591722379107572}

params = dict()
params.update(xgb_default_params)
params.update(best_param)




model = xgb.XGBModel(**xgb_default_params)


model.fit(X_train,y_train,early_stopping_rounds = early_stop, eval_metric ='mlogloss',verbose = True,eval_set = [(X_train,y_train),(X_test,y_test)])

result = model.evals_result()
print(result)




'''
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
X_train = pd.read_csv("X_train.csv")
oof_train = np.zeros((X_train.shape[0]))
oof_test = np.zeros((X_test.shape[0], n_splits))

eval_log = np.zeros((n_splits,2))


i = 0

for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):

    X_tr = X_train.iloc[train_idx, :]
    X_val = X_train.iloc[valid_idx, :]

    y_tr = X_tr['AdoptionSpeed'].values
    X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

    y_val = X_val['AdoptionSpeed'].values
    X_val = X_val.drop(['AdoptionSpeed'], axis=1)

    d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
    d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)


    



    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist, 
                        early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

    print(model.evals_result)
    
    valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

    oof_train[valid_idx] = valid_pred
    oof_test[:, i] = test_pred

    i += 1
    
    
    
    
    

optR = OptimizedRounder()
optR.fit(oof_train, X_train['AdoptionSpeed'].values)
coefficients = optR.coefficients()
valid_pred = optR.predict(oof_train, coefficients)
qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
print("QWK = ", qwk)


coefficients_ = coefficients.copy()
coefficients_[0] = 1.66
coefficients_[1] = 2.13
coefficients_[3] = 2.85
train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
print(f'train pred distribution: {Counter(train_predictions)}')
test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
print(f'test pred distribution: {Counter(test_predictions)}')
'''