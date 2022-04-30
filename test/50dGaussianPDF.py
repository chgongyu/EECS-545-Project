from tuning import BayesOpt_KSD
from gpytorch.kernels import MaternKernel
from botorch.acquisition.analytic import UpperConfidenceBound
from ax import (
    ParameterType,
    RangeParameter,
    SearchSpace,
)
import numpy as np
from scipy.linalg import block_diag

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=-0.5, upper=0.5
        )
        for i in range(50)
    ]
)

n = 25
a = np.linalg.inv(np.array([[1., 0.9], [0.9, 1.]]))
precision = block_diag(*([a] * n))

def gaussianPDF(param):
    try:
      param = param.reshape((-1, 1))
    except:
      param = np.array(list(param.values())).reshape((-1, 1))
    val = np.exp(- param.T @ precision @ param)
    return -val[0, 0]

inits = (25, 25)
iters = 500
runs = 10

GLOBAL_Y_COLLECTOR = np.zeros((runs, sum(inits)+iters))

test_case = BayesOpt_KSD(search_space, gaussianPDF)

for run in range(runs):
  test_case.optimize(MaternKernel, UpperConfidenceBound, n_init=inits, n_iter=iters, alpha=0.4)
  GLOBAL_Y_COLLECTOR[run] = test_case.Y_COLLECTOR
