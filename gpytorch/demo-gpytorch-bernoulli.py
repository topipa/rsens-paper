import torch
import gpytorch
import math
from matplotlib import pyplot as plt

import rsens_gpytorch

## simulate data
torch.manual_seed(1)
train_x = torch.rand(200)

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))
def E(x):
    return 5*torch.sin(10*x)
train_f = E(train_x)
train_f = train_f - train_f.mean()
train_y = torch.bernoulli(sigmoid(train_f))

## fit GP model
class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, likelihood):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

likelihood = gpytorch.likelihoods.BernoulliLikelihood()
model = GPClassificationModel(train_x, likelihood)

training_iter = 500
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
obj_fn = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -obj_fn(output, train_y)
    loss.backward()
    # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

model.eval()
likelihood.eval()

## evaluate R-sens

test_x = torch.linspace(-0.25, 1.25, 1000)
dE_dx = rsens_gpytorch.absderiv(model, test_x)
RS = rsens_gpytorch.rsens(model, test_x)


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label = 'p(y = 1)')
    ax1.scatter(train_x.numpy(), train_y.numpy(), c='k', s = 6, marker = 'o')
    ax1.legend()
    ax1.set_ylabel('$y$')

    ax2.plot(test_x,2*dE_dx, c = 'k', label = '$ |dE/dx$|')
    ax2.plot(test_x,RS, c = 'red', label = 'R-sens')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('sensitivity')
    ax2.legend()

    plt.show()

