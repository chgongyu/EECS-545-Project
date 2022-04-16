
from tuning.bayes import BayesOpt_KSD

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
