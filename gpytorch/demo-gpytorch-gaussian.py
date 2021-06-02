import torch
import gpytorch
import math
from matplotlib import pyplot as plt

import rsens_gpytorch

## simulate data
torch.manual_seed(8)
train_x = torch.rand(20)

def E(x):
    return 20*torch.sin(10*x)
train_y = E(train_x) + torch.randn(train_x.size())
train_y = train_y - train_y.mean()

## fit GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

model.eval()
likelihood.eval()

## evaluate R-sens

test_x = torch.linspace(-0.25, 1.25, 1000)
dE_dx = rsens_gpytorch.absderiv(model, test_x)
RS = rsens_gpytorch.rsens(model, test_x)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    lower, upper = observed_pred.confidence_region()


    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    ax1.scatter(train_x.numpy(), train_y.numpy(), c='k', s = 6, marker = 'o')
    ax1.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, color = 'b', edgecolor = 'none')
    ax1.set_ylabel('$y$')

    ax2.plot(test_x,dE_dx, c = 'k', label = '$ |dE/dx$|')
    ax2.plot(test_x,RS, c = 'red', label = 'R-sens')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('sensitivity')

    plt.legend()

    plt.show()

