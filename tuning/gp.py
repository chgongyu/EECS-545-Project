import torch
from gpytorch.kernels.kernel import AdditiveKernel, Kernel
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.containers import TrainingData
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, ProductStructureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ExactGP
from typing import Iterable, List

class CliqueAdditiveGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, cliques, kernelType):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean() # or ZeroMean

        # construct an additive kernel from product of 1D kernels in each clique
        for idx, cliq in enumerate(cliques):
          if idx == 0:
            self.covar_module = ScaleKernel(
                                            ProductStructureKernel(
                                                kernelType(),
                                                num_dims=train_X.shape[-1],
                                                active_dims=torch.tensor(cliq))
                                            )
          else:
            self.covar_module += ScaleKernel(
                                            ProductStructureKernel(
                                                kernelType(),
                                                num_dims=train_X.shape[-1],
                                                active_dims=torch.tensor(cliq))
                                            )

        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls,
                         training_data: TrainingData,
                         cliques: Iterable[List],
                         kernelType: Kernel,
                         **kwargs):
        r"""Construct kwargs for the `SimpleCustomGP` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            cliques: `Iterable[list]` with single element a list of clique
            kernelType: gpytorch `Kernel` class
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X,
                "train_Y": training_data.Y,
                "cliques": cliques,
                "kernelType": kernelType
               }
