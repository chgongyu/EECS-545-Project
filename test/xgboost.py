from tuning import BayesOpt_KSD
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import pandas as pd
from ax import (
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from gpytorch.kernels import MaternKernel
from botorch.acquisition.analytic import UpperConfidenceBound

# XGB Test Case
# from google.colab import drive
# drive.mount('/content/drive')
# %cd '/content/drive/My Drive/545'
df = pd.read_csv(r'Churn_Modelling.csv')
df.head()
# Getting the independent features
X = df.iloc[:, 3:-1]
# Getting the dependent feature
y = df.iloc[:, -1]

# Dummy variable for 'Geography' column
geography = pd.get_dummies(X['Geography'], drop_first=True)
# Dummy variable for 'Gender' column
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Dropping the original 'Geography' and 'Gender' columns
X = X.drop(['Geography', 'Gender'], axis=1)

# Adding the dummy columns to the dataset
X = pd.concat([X, geography, gender], axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

global_parms = {"booster": "gbtree",
                "missing": None,
                "n_estimators": 100,
                "n_jobs": 1,
                "objective": 'binary:logistic',
                "random_state": 545,
                "scale_pos_weight": 1,
                "verbosity": 1,
                "eval_metric": 'logloss'
                }

params_space = {
    "base_score": (ParameterType.FLOAT, 0.1, 0.9),
    "colsample_bylevel": (ParameterType.FLOAT, 0.1, 1),
    "colsample_bynode": (ParameterType.FLOAT, 0.1, 1),
    "colsample_bytree": (ParameterType.FLOAT, 0.1, 1),
    "learning_rate": (ParameterType.FLOAT, 0.1, 1),
    "max_delta_step": (ParameterType.INT, 0, 10),  # integer
    "max_depth": (ParameterType.INT, 1, 20),  # integer
    "min_child_weight": (ParameterType.FLOAT, 1, 20),
    "gamma": (ParameterType.FLOAT, 0.1, 10),
    "subsample": (ParameterType.FLOAT, 0.1, 1),
    "reg_lambda": (ParameterType.FLOAT, 1, 9),
    "reg_alpha": (ParameterType.FLOAT, 1, 9),
}

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=key, parameter_type=val[0], lower=val[1], upper=val[2]
        )
        for key, val in params_space.items()
    ]
)


def run_xgb(params,
            # X_train,
            # y_train,
            fold=5):
    # X_train = X_train
    # y_train = y_train
    param_all = global_parms.copy()
    param_all.update(params)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv_result = xgb.cv(param_all, dtrain, num_boost_round=100, nfold=fold)
    # print(cv_result)
    return cv_result['test-logloss-mean'].iloc[-1]

def test_xgb(params,
             # X_train,
             # y_train,
             # fold = 10
             ):
    # X_train = X_train
    # y_train = y_train
    param_all = global_parms.copy()
    param_all.update(params)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(param_all, dtrain, num_boost_round=100)
    y_pred = model.predict(dtest)
    return {'logloss': log_loss(y_test, y_pred),
            'acc': accuracy_score(y_test, np.round(y_pred))
            }


inits = (30, 20)
iters = 200
runs = 20

GLOBAL_Y_COLLECTOR = np.zeros((runs, sum(inits) + iters))
GLOBAL_TEST_LOGLOSS = np.zeros((runs,))
GLOBAL_TEST_ACC = np.zeros((runs,))

test_case = BayesOpt_KSD(search_space, run_xgb)

for run in range(runs):
    test_case.optimize(MaternKernel, UpperConfidenceBound, n_init=inits, n_iter=iters, alpha=0.4)

    best_params = test_case.get_best_params()
    prediction = test_xgb(best_params)
    
    GLOBAL_TEST_LOGLOSS[run] = prediction['logloss']
    GLOBAL_TEST_ACC[run] = prediction['acc']
    GLOBAL_Y_COLLECTOR[run] = test_case.Y_COLLECTOR
