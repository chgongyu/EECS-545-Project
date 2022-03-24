import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from fanova import fANOVA
# import fanova.visualizer
# import os
from collections import OrderedDict
import itertools as it
import multiprocessing as mp

class HyperImpt:
    def __init__(self, X, y, space):
        self.cs = self.__init_cs__(space)
        self.f = fANOVA(X=X, Y=y, config_space=self.cs)

    def __init_cs__(self, space):
        cs = ConfigurationSpace()
        # TODO: Vectorize the hyperparameter adding process
        for key, val in space.items():
            cs.add_hyperparameter(UniformFloatHyperparameter(key, lower=val[0], upper=val[1]))
        return cs

    def get_most_important_marginals(self, params=None, n=10):
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
            dimensions = range(self.f.n_dims)
        else:
            if type(params[0]) == str:
                idx = []
                for i, param in enumerate(params):
                    idx.append(self.f.cs.get_idx_by_hyperparameter_name(param))
                dimensions = idx

            else:
                dimensions = params
        # pairs = it.combinations(dimensions,2)
        pairs = [x for x in it.combinations(dimensions, 1)]
        if params:
            n = len(list(pairs))
        for combi in pairs:
            marginal_performance = self.f.quantify_importance(combi)
            tot_imp = marginal_performance[combi]['individual importance']
            combi_names = self.f.cs_params[combi[0]].name
            marginals.append((tot_imp, combi_names))

        marginal_performance = sorted(marginals, reverse=True)

        for marginal, p in marginal_performance[:n]:
            tot_imp_dict[p] = marginal

        return tot_imp_dict


    def get_most_important_individual_pariwise(self, params=None, n=10):
        tot_imp_dict = OrderedDict()
        marginals = list(self.get_most_important_marginals(self.f, n=2).items()) + \
                    list(self.f.get_most_important_pairwise_marginals(n=2).items())
        performance = sorted(marginals, key=lambda item: item[1], reverse=True)

        for p, marginal in performance[:n]:
            tot_imp_dict[p] = marginal

        return tot_imp_dict

    def run(self, n_impt=5):
        res = self.get_most_important_individual_pariwise(n=n_impt)
        print(res)