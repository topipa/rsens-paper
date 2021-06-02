import torch
import rsens


def E(x):
        # return torch.log(1.0 + torch.exp(2.5 * x))
        # return x
        # return x**2
        # return x**3
        # return x + torch.cos(9*x)
        # return torch.sin(9*x)
        return x * torch.exp(-x)
        # return torch.exp(-5*x**2)

def V(x):
        return 0.03 * (1 + 5 * (x-0.2)**2)

rsens.rsens_demo_plot(E, V, plot_separate_terms = True)
