import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import rsens_gpytorch

## simulate data
torch.manual_seed(8)

N = 300
D = 10
train_x = torch.randn((N,D))
def E(x):
    return torch.log(1.0 + torch.exp(2.5 * x))
    # return x
train_f = torch.zeros(N)
for i in range(D):
    train_f = train_f + (i+1)*E(train_x[:,i])
train_f = train_f/train_f.std()
train_y = train_f + 0.2*torch.randn(train_f.size())
train_y = train_y - train_y.mean()

## fit GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveStructureKernel(base_kernel = gpytorch.kernels.RBFKernel(), num_dims = D, active_dims = torch.arange(D)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
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

## evaluate R-sens

test_x = train_x.clone()
RS = rsens_gpytorch.rsens(model, test_x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(torch.arange(D).numpy()+1, RS.detach().mean(axis=0), 'k.-', markersize = 10)
ax1.set_xlabel('variable')
ax1.set_ylabel('importance')

plt.show()

