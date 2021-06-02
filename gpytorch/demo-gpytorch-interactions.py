import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import rsens_gpytorch

## simulate data
torch.manual_seed(8)
N = 300
D = 4
train_x = torch.randn((N,D))

train_f = train_x[:,2]*train_x[:,3] + 2*train_x[:,0]*train_x[:,3]
train_f = train_f/train_f.std()

train_y = train_f + 0.2*torch.randn(train_f.size())
train_y = train_y - train_y.mean()

## fit GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveStructureKernel(base_kernel = gpytorch.kernels.RBFKernel(), num_dims = D, active_dims = torch.arange(D)))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

training_iter = 100

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

## evaluate R-sens2

test_x = train_x.clone()
RS2 = rsens_gpytorch.rsens2(model, test_x)


plt.matshow(RS2.mean(axis=0).detach().numpy(), cmap = 'Blues')
plt.xlabel('variable #1')
plt.ylabel('variable #2')
plt.colorbar()
plt.show()

