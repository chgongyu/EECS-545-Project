import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from ax import (
    Metric,
    Data,
)
from typing import Callable, Dict, List,  Optional


class CustomMetric(Metric):
    """A metric defined by a custom input model.
    """

    def __init__(
        self,
        name: str,
        param_names: List[str],
        evaluate: Callable[[Dict], float],
        lower_is_better: Optional[bool] = None,
    ) -> None:
        """
        Args:
            name: Name of the metric
            param_names: An ordered list of names of parameters to be passed
                to the `evaluate` function.
            evaluate: A function that evaluates the performance of the parameters
            lower_is_better: Flag for metrics which should be minimized.
        """
        self.param_names = param_names
        self.evaluate = evaluate
        super().__init__(name=name, lower_is_better=lower_is_better)

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                # "mean": self.evaluate(params),
                "mean": self.evaluate(np.fromiter(params.values(), dtype=float)),
                "sem": 0.0,
            })
        return Data(df=pd.DataFrame.from_records(records))


def s_main_i(x, xp):
    N = x.shape[0]
    M = xp.shape[0]

    dist1 = cdist(x, x, 'sqeuclidean')
    dist1 = dist1[np.triu_indices_from(dist1, k=1)].reshape(-1)

    dist2 = cdist(xp, xp, 'sqeuclidean')
    dist2 = dist2[np.triu_indices_from(dist2, k=1)].reshape(-1)

    dist3 = cdist(x, xp, 'sqeuclidean').reshape(-1)

    gamma = 1 / np.median(np.sqrt(dist1)) ** 2 / 2

    term1 = np.sum(np.exp(-gamma * dist1)) / N / (N - 1)
    term2 = np.sum(np.exp(-gamma * dist2)) / M / (M - 1)
    term3 = -2 * np.sum(np.exp(-gamma * dist3)) / N / M
    output = 2 * (term1 + term2) + term3
    # print(output)
    return output if output >= 0 else 0


def s_int_i_j(x_i, xp_i, x_j, xp_j):
    N = x_i.shape[0]
    M = xp_i.shape[0]

    dist1_i = cdist(x_i, x_i, 'sqeuclidean')
    dist1_i = dist1_i[np.triu_indices_from(dist1_i, k=1)].reshape(-1)
    dist1_j = cdist(x_j, x_j, 'sqeuclidean')
    dist1_j = dist1_j[np.triu_indices_from(dist1_j, k=1)].reshape(-1)

    dist2_i = cdist(xp_i, xp_i, 'sqeuclidean')
    dist2_i = dist2_i[np.triu_indices_from(dist2_i, k=1)].reshape(-1)
    dist2_j = cdist(xp_j, xp_j, 'sqeuclidean')
    dist2_j = dist2_j[np.triu_indices_from(dist2_j, k=1)].reshape(-1)

    dist3_i = cdist(x_i, xp_i, 'sqeuclidean').reshape(-1)
    dist3_j = cdist(x_j, xp_j, 'sqeuclidean').reshape(-1)

    gamma_i = 1 / np.median(np.sqrt(dist1_i)) ** 2 / 2
    gamma_j = 1 / np.median(np.sqrt(dist1_j)) ** 2 / 2

    kernel1 = (1 + np.exp(-gamma_i * dist1_i)) * (1 + np.exp(-gamma_j * dist1_j))
    kernel2 = (1 + np.exp(-gamma_i * dist2_i)) * (1 + np.exp(-gamma_j * dist2_j))
    kernel3 = (1 + np.exp(-gamma_i * dist3_i)) * (1 + np.exp(-gamma_j * dist3_j))
    term1 = np.sum(kernel1) / N / (N - 1)
    term2 = np.sum(kernel2) / M / (M - 1)
    term3 = -2 * np.sum(kernel3) / N / M

    # main_i1 = np.sum(np.exp(-gamma_i * (dist1_i)))/ N / (N - 1)
    # main_i2 = np.sum(np.exp(-gamma_i * (dist2_i)))/ M / (M - 1)
    # main_i3 = -2 * np.sum(np.exp(-gamma_i * (dist3_i))) / N / M
    # main_i = 2*(main_i1 + main_i2) + main_i3
    # # main_i = main_i if main_i >= 0 else 0

    # main_j1 = np.sum(np.exp(-gamma_j * (dist1_j)))/ N / (N - 1)
    # main_j2 = np.sum(np.exp(-gamma_j * (dist2_j)))/ M / (M - 1)
    # main_j3 = -2 * np.sum(np.exp(-gamma_j * (dist3_j))) / N / M
    # main_j = 2*(main_j1 + main_j2) + main_j3
    # main_j = main_j if main_j >= 0 else 0

    output = 2 * (term1 + term2) + term3
    # print(output)
    return output if output >= 0 else 0


def s_main(X, Xp):
    main = np.zeros((X.shape[1],))
    for i in range(X.shape[1]):
        main[i] = s_main_i(X[:, [i]], Xp[:, [i]])
    return main


def s_int(X, Xp):
    interaction = np.zeros((X.shape[1], X.shape[1]))
    main = s_main(X, Xp)
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            interaction[i][j] = s_int_i_j(X[:, [i]], Xp[:, [i]], X[:, [j]], Xp[:, [j]])
    np.fill_diagonal(interaction, main)
    return interaction


def s_total(X, xp):
    total = np.zeros((X.shape[1],))
    interaction = s_int(X, xp)
    for i in range(X.shape[1]):
        total[i] = np.sum(interaction[i])
    return total