import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from fanova import fANOVA
# import fanova.visualizer
# import os
from collections import OrderedDict
import itertools as it
import multiprocessing as mp

#%%
# # path = os.path.dirname(os.path.realpath(__file__))

# # get sample data from online lda
# X = np.loadtxt('online_lda_features.csv', delimiter=",")
# Y = np.loadtxt('online_lda_responses.csv', delimiter=",")

# # setting up config space:
# param_file = 'param-file.txt'
# f = open(param_file, 'rb')

# cs = ConfigurationSpace()
# for row in f:
#     cs.add_hyperparameter(UniformFloatHyperparameter("%s" %row[0:4].decode('utf-8'), float(row[6:9]), float(row[10:13]), float(row[18:21])))
# param = cs.get_hyperparameters()

# #%%
# # create an instance of fanova with data for the random forest and the configSpace
# f = fANOVA(X = X, Y = Y, config_space = cs)

# #%%
# # marginal for first parameter
# p_list = (0, )
# res = f.quantify_importance(p_list)
# print(res)

# p2_list = ('Col1', 'Col2')
# res2 = f.quantify_importance(p2_list)
# print(res2)
# p2_list = ('Col0', 'Col2')
# res2 = f.quantify_importance(p2_list)
# print(res2)
# p2_list = ('Col1', 'Col0')
# res2 = f.quantify_importance(p2_list)
# print(res2)

# # getting the most important pairwise marginals sorted by importance
# best_p_margs = f.get_most_important_pairwise_marginals(n=3)
# print(best_p_margs)

# triplets = f.get_triple_marginals(['Col0','Col1', 'Col2'])
# print(triplets)

# #%%
# # visualizations:
# # directory in which you can find all plots
# plot_dir = os.getcwd()
# # first create an instance of the visualizer with fanova object and configspace
# vis = fanova.visualizer.Visualizer(f, cs, plot_dir)
# # generating plot data for col0
# mean, std, grid = vis.generate_marginal(0)

# # creating all plots in the directory
# vis.create_all_plots()
# #vis.create_most_important_pairwise_marginal_plots(plot_dir, 3)

#%%
def get_most_important_marginals(f, params=None, n=10):
      """
      Returns the n most important pairwise marginals from the whole ConfigSpace
          
      Parameters
      ----------
      params: list of strings or ints
          If specified, limit analysis to those parameters. If ints, interpreting as indices from ConfigurationSpace
      n: int
           The number of most relevant pairwise marginals that will be returned
        
      Returns
      -------
      list: 
           Contains the n most important pairwise marginals
      """
      tot_imp_dict = OrderedDict()
      marginals = []
      if params is None:
          dimensions = range(f.n_dims)
      else:
          if type(params[0]) == str:
              idx = []
              for i, param in enumerate(params):
                  idx.append(f.cs.get_idx_by_hyperparameter_name(param))
              dimensions = idx

          else:
              dimensions = params
      # pairs = it.combinations(dimensions,2)
      pairs = [x for x in it.combinations(dimensions, 1)]
      if params:
          n = len(list(pairs))
      for combi in pairs:
          marginal_performance = f.quantify_importance(combi)
          tot_imp = marginal_performance[combi]['individual importance']
          combi_names = f.cs_params[combi[0]].name
          marginals.append((tot_imp, combi_names))

      marginal_performance = sorted(marginals, reverse=True)

      for marginal, p in marginal_performance[:n]:
          tot_imp_dict[p] = marginal

      return tot_imp_dict

def get_most_important_individual_pariwise(f, params=None, n=10):
    tot_imp_dict = OrderedDict()
    marginals = list(get_most_important_marginals(f, n=2).items()) + \
        list(f.get_most_important_pairwise_marginals(n=2).items())
    performance = sorted(marginals, key=lambda item: item[1], reverse=True)
    
    for p, marginal in performance[:n]:
          tot_imp_dict[p] = marginal

    return tot_imp_dict

#%% 
# res = get_most_important_individual_pariwise(f, n=4)
# print(res)
from timeit import default_timer as timer


#%% 
from sklearn.model_selection import ParameterSampler, train_test_split
import pandas as pd
import xgboost as xgb
X_train_non_null = pd.read_csv("X_train.csv")
#X_test_non_null = pd.read_csv("X_test.csv")

y = X_train_non_null['AdoptionSpeed']
X = X_train_non_null.drop(['AdoptionSpeed'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
dtrain = xgb.DMatrix(X_train, label = y_train)

#%%
seed = 1337
rng = np.random.RandomState(seed)
space = {
          'eta':(0.01,0.2),
          'gamma':(0,25),
          'max_depth':(2,18),
          'min_child_weight':(0,20),
          'max_delta_step':(0,10),
          'subsample':(0.5,1),
          'colsample_bytree':(0.5,1),
          'reg_alpha' :(40,180),
          'reg_lambda' :(0,1)
          # , 'n_estimator' :(80,120)
          }

def tune_xgb(param, nfold=5):
    xgb_param = {
              # 'eval_metric': 'merror',
              'eval_metric': 'rmse',
              'max_depth': int(param['max_depth']),
              'subsample': param['subsample'],
              'eta': param['eta'],
              'gamma': param['gamma'],
              'min_child_weight': param['min_child_weight'],
              'max_delta_step': param['max_delta_step'],
              'colsample_bytree':param['colsample_bytree'],
              'reg_alpha' : param['reg_alpha'],
              'reg_lambda' :param['reg_lambda'],
              'seed': seed,
              # 'num_class': 5,
              'silent': True,
              'verbosity': 0
              # 'n_estimator' : param['n_estimator']
              }   
    cv_result = xgb.cv(xgb_param, dtrain, num_boost_round=70, nfold=nfold)
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


param_list = list(ParameterSampler(space, n_iter=50, random_state=rng))
rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items()) for d in param_list]


#%%
responses = []

def collect_results(result):
    global responses
    responses.append(result)

def train_xgb(param):
    return (param, tune_xgb(param, nfold=3))
        
def main():
    pool = mp.Pool(6)
    start = timer()
    # responses = [pool.apply(train_xgb, args=(param, )) for param in rounded_list]
    for param in rounded_list:
        pool.apply_async(train_xgb, args=(param, ), callback=collect_results)
        # responses.append(train_xgb(param))
    
    pool.close()
    pool.join()
    print(timer()- start)
    # print(responses)
    # X = pd.DataFrame.from_dict(rounded_list)
    # y = pd.Series(responses)
    
    # cs = ConfigurationSpace()
    # for key, val in space.items():
    #     if key in ['max_depth']:
    #         cs.add_hyperparameter(UniformIntegerHyperparameter(key, lower=val[0], upper=val[1]))
    #     else:
    #         cs.add_hyperparameter(UniformFloatHyperparameter(key, lower=val[0], upper=val[1]))
    
    # f = fANOVA(X = X, Y = y, config_space = cs)

    # res = get_most_important_individual_pariwise(f, n=5)
    # print(res)

if __name__ == '__main__':
    main()
    