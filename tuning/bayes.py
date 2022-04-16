import networkx as nx
from gp import *
from utils import *
from gpytorch.kernels.kernel import AdditiveKernel, Kernel
from botorch.acquisition import AcquisitionFunction
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax import (
    ComparisonOp,
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    SearchSpace,
    Experiment,
    OutcomeConstraint,
    OrderConstraint,
    SumConstraint,
    OptimizationConfig,
    Objective,
    Metric,
    Data,
    Runner
)
from typing import Callable, Dict, Tuple


class BayesOpt_KSD():
    # experiment status labels
    UNKNOWN = "uninitialized"
    CONSTRUCTED = "constructed"
    COMPLETED = "completed"

    def __init__(self, search_space: SearchSpace,
                 evaluate: Callable[[Dict], float],
                 minimize: bool = True,
                 ):
        self.search_space = search_space
        self.param_names = search_space.parameters.keys()
        self.evaluate = evaluate
        self.minimize = minimize
        self.sobol = Models.SOBOL(search_space=search_space)
        self.optimization_config = OptimizationConfig(
            objective=Objective(
                metric=CustomMetric(name="loss",
                                    param_names=self.param_names,
                                    evaluate=evaluate
                                    ),
                minimize=minimize)
        )

        self.cliques = [list(range(len(self.param_names)))]
        self.alpha = 0.5  # thresholding quantile
        self.exp_status = BayesOpt_KSD.UNKNOWN

    def gen_bo(self, kernelType, acquisition):
        return Models.BOTORCH_MODULAR(
            experiment=self.exp,
            data=self.exp.fetch_data(),
            surrogate=Surrogate(CliqueAdditiveGP,
                                model_options={'cliques': self.cliques,
                                               'kernelType': kernelType}),
            botorch_acqf_class=acquisition,
        )

    def _thresholding(self):
        X, Y = self.collect_data()
        if self.minimize:
            q_a = np.quantile(Y, self.alpha)
            return X, X[Y <= q_a]
        else:
            q_a = np.quantile(Y, 1 - self.alpha)
            return X, X[Y >= q_a]

    def get_sobol_weights(self):
        X, Xp = self._thresholding()
        weights = s_total(X, Xp)
        print(weights)
        if np.max(weights) <= 1e-8:
            return np.ones(weights.shape)
        weights /= np.max(weights)
        print(weights)
        return weights

    def get_cliques(self):
        prop = 0.9
        X, Xp = self._thresholding()
        # print(X.shape, Xp.shape)
        interact = s_int(X, Xp)
        print(interact)
        tau = 0.1 * np.max(interact)
        interact[interact < tau] = 0
        interact /= np.sum(np.triu(interact))
        # print(interact)
        # print(interact)
        n_nodes = X.shape[1]
        G = nx.Graph()
        for src in range(n_nodes):
            G.add_node(src)
            for dst in range(n_nodes):
                if interact[src, dst] != 0:
                    G.add_edge(src, dst, weight=interact[src, dst])

        cliques = list(nx.find_cliques(G))
        weights = []
        output = []
        isotropic = []
        total_weight = 0
        for c in cliques:
            indiv_weight = 0
            for src in c:
                for dst in c:
                    if src <= dst:
                        indiv_weight += interact[src, dst]
            total_weight += indiv_weight
            weights.append({'clique': c, 'weight': indiv_weight})
        weights.sort(reverse=True, key=lambda a: a['weight'])

        c_weight = 0
        prop *= total_weight
        for item in weights:
            prev_weight = c_weight
            c_weight += item['weight']
            if c_weight < prop or (prev_weight < prop and c_weight > prop):
                output.append(item['clique'])
            else:
                isotropic += item['clique']
        if len(isotropic) > 0:
            output.append(isotropic)

        print(weights)
        print(output)
        return output

    def collect_data(self):
        if self.exp_status is BayesOpt_KSD.CONSTRUCTED:
            # print(self.X_COLLECTOR[:self.exp.num_trials])
            # print(self.exp.fetch_data().df['mean'])
            return self.X_COLLECTOR[:self.exp.num_trials], self.exp.fetch_data().df['mean'].to_numpy()
        elif self.exp_status is BayesOpt_KSD.COMPLETED:
            return self.X_COLLECTOR, self.Y_COLLECTOR
        else:
            print("No experiment has been performed.")

    def _run_trial(self, generator_run):
        trial = self.exp.new_trial(generator_run=generator_run)
        trial.run()
        param_data = list(trial.arm.parameters.values())
        self.X_COLLECTOR[self.exp.num_trials - 1] = param_data
        trial.mark_completed()

    class _DummyRunner(Runner):
        def run(self, trial):
            trial_metadata = {"name": str(trial.index)}
            return trial_metadata

    def optimize(self, kernelType: Kernel,
                 acquisition: AcquisitionFunction,
                 n_init: Tuple[int, int] = (10, 5),
                 n_iter: int = 10,
                 alpha=0.25
                 ):
        self.NUM_SOBOL_TRIALS = n_init[0]
        self.NUM_WEIGHTED_SOBOL_TRIALS = n_init[1]
        self.NUM_BOTORCH_TRIALS = n_iter
        self.NUM_TRIALS = sum(n_init) + n_iter
        self.alpha = alpha
        self.X_COLLECTOR = np.zeros((self.NUM_TRIALS, len(self.param_names)))
        self.Y_COLLECTOR = np.zeros(self.NUM_TRIALS)

        self.exp = Experiment(
            name="hyper opt",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            runner=self._DummyRunner()
        )
        self.exp_status = BayesOpt_KSD.CONSTRUCTED

        # Phase 0: Sobol sequence
        for i in range(self.NUM_SOBOL_TRIALS):
            self._run_trial(self.sobol.gen(n=1))

        weights = self.get_sobol_weights()

        # Phase 1: weighted Sobol sequence
        for i in range(self.NUM_WEIGHTED_SOBOL_TRIALS):
            generator_run = self.sobol.gen(n=1)
            p = np.random.rand()
            for idx, param in enumerate(self.param_names):
                if p > weights[idx]:
                    generator_run.arms[0].parameters[param] = self.X_COLLECTOR[self.exp.num_trials - 1][idx]
            # fixed_features = {idx: self.X_COLLECTOR[self.exp.num_trials-1][idx]
            #                     for idx in range(len(self.param_names))
            #                     if p > weights[idx]
            #                  }
            self._run_trial(generator_run)

        self.cliques = self.get_cliques()

        # Phase 2: Bayesian optimization
        for i in range(self.NUM_BOTORCH_TRIALS):
            gpei = self.gen_bo(kernelType, acquisition)
            self._run_trial(gpei.gen(n=1))

        self.Y_COLLECTOR = self.exp.fetch_data().df['mean'].to_numpy()

        self.exp_status = BayesOpt_KSD.COMPLETED
