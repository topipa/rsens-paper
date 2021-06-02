import torch
import rsens

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

def p(x):
        # return sigmoid(torch.log(1.0 + torch.exp(2.5 * x)))
        # return sigmoid(5*x)
        # return sigmoid(5*x**2)
        # return sigmoid(5*x**3)
        # return sigmoid(2*(x + torch.cos(9*x)))
        return sigmoid(3*torch.sin(9*x))
        # return sigmoid(3*x * torch.exp(-x))
        # return sigmoid(3*torch.exp(-5*x**2))

rsens.rsens_demo_plot_bernoulli(p, plot_separate_terms = False)



