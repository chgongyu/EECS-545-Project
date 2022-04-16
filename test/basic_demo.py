from tuning import BayesOpt_KSD

import numpy as np
from ax import (
    ComparisonOp,
    ParameterType,
    RangeParameter,
    # ChoiceParameter,
    FixedParameter,
    SearchSpace,
}
from ax.utils.measurement.synthetic_functions import hartmann6
from gpytorch.kernels import MaternKernel
from botorch.acquisition.analytic import ExpectedImprovement

from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for i in range(6)
    ]
)

test_case = BayesOpt_KSD(search_space, hartmann6)

test_case.optimize(MaternKernel, ExpectedImprovement, n_init=(50, 10), n_iter=100, alpha=0.4)

# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
# optimization runs, so we wrap out best objectives array in another array.
objective_means = np.array([[trial.objective_mean for trial in test_case.exp.trials.values()]])
best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(objective_means, axis=1),
        optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
)
render(best_objective_plot)