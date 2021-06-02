import numpy as np
import torch
from matplotlib import pyplot as plt

def rsens_demo_plot(E, V, plot_separate_terms = False):

    xx = torch.linspace(-1.0,1.0, steps=1000, requires_grad=True)
    assert (V(xx) > 0.0).all().item(), "V(x) must be positive"

    def dE(xx):
        Ex = E(xx)
        external_grad = torch.ones(xx.shape[0])
        dEx = torch.autograd.grad(Ex,xx, grad_outputs=external_grad)
        return dEx

    def dV(xx):
        Vx = V(xx)
        external_grad = torch.ones(xx.shape[0])
        dVx = torch.autograd.grad(Vx,xx, grad_outputs=external_grad)
        return dVx

    def term1(Vx, dEx):
        return dEx**2/Vx

    def term2(Vx, dVx):
        return dVx**2/(2*Vx**2)

    Ex = E(xx)
    dEx = dE(xx)[0].numpy()
    Vx = V(xx)
    dVx = dV(xx)[0].numpy()

    x = xx.detach().numpy()
    Ex = Ex.detach().numpy()
    Vx = Vx.detach().numpy()

    t1 = term1(Vx, dEx)
    t2 = term2(Vx, dVx)
    RS = np.sqrt(t1 + t2)
    RS_scale = 1.0/np.max(RS)
    dEx_scale = 1.0/np.max(np.abs(dEx))

    figw =  7
    fig = plt.figure(figsize=(figw,0.65*figw))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax.plot(x,Ex, color = '#1000ba')
    ax.fill_between(x,Ex - np.sqrt(Vx),Ex + np.sqrt(Vx),alpha=0.1,color='blue',lw=0, edgecolor = 'none', label = "$p(y | x)$")
    ax.fill_between(x,Ex - 2*np.sqrt(Vx),Ex + 2*np.sqrt(Vx),alpha=0.1,color='blue',lw=0, edgecolor = 'none')
    ax.fill_between(x,Ex - 3*np.sqrt(Vx),Ex + 3*np.sqrt(Vx),alpha=0.1,color='blue',lw=0, edgecolor = 'none')

    ax2.plot(x,dEx_scale*np.abs(dEx), color = 'black', label = '$ \\left| \\frac{\partial \mathrm{E}[y | x] }{ \partial x} \\right|$')
    ax2.plot(x,RS_scale*RS, color = 'red', label = 'R-sens')

    if (plot_separate_terms):
        ax2.plot(x,RS_scale*np.sqrt(t1),'--', color = 'red', label = 'term1')
        ax2.plot(x,RS_scale*np.sqrt(t2),':', color = 'red', label = 'term2')

    plt.subplots_adjust(hspace = 0.0)

    ax.tick_params(direction='in',which='both', top=True, right=True,zorder=1)
    ax2.tick_params(direction='in',which='both', top=True, right=True,zorder=1)

    ax.set_xticks([])
    ax2.set_ylim([0,1.05])

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('sensitivity', labelpad = 10)
    ax.set_ylabel('$y$')

    ax.set_xlim([x.min(),x.max()])
    ax2.set_xlim([x.min(),x.max()])

    ax.legend()
    ax2.legend()

    plt.show()


def rsens_demo_plot_bernoulli(p, plot_separate_terms = False):

    xx = torch.linspace(-1.0,1.0, steps=1000, requires_grad=True)
    assert (p(xx) > 0.0).all().item(), "p(x) must be positive"
    assert (p(xx) < 1.0).all().item(), "p(x) must be less than 1"

    def dp(xx):
        px = p(xx)
        external_grad = torch.ones(xx.shape[0])
        dpx = torch.autograd.grad(px,xx, grad_outputs=external_grad)
        return dpx

    def term1(px, dpx):
        return dpx**2/(px*(1.0-px))

    px = p(xx)
    dpx = dp(xx)[0].numpy()

    x = xx.detach().numpy()
    px = px.detach().numpy()

    t1 = term1(px, dpx)
    RS = np.sqrt(t1)
    RS_scale = 1.0/np.max(RS)
    dpx_scale = 1.0/np.max(np.abs(dpx))

    figw =  7
    fig = plt.figure(figsize=(figw,0.65*figw))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax.plot(x,px, color = '#1000ba', label = '$\pi = p(y = 1)$')

    ax2.plot(x,dpx_scale*np.abs(dpx), color = 'black', label = '$ \\left| \\frac{\partial \pi }{ \partial x} \\right|$')
    ax2.plot(x,RS_scale*RS, color = 'red', label = 'R-sens')

    plt.subplots_adjust(hspace = 0.0)

    ax.tick_params(direction='in',which='both', top=True, right=True,zorder=1)
    ax2.tick_params(direction='in',which='both', top=True, right=True,zorder=1)

    ax.set_xticks([])
    ax2.set_ylim([0,1.05])

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('sensitivity', labelpad = 10)
    ax.set_ylabel('$y$')

    ax.set_xlim([x.min(),x.max()])
    ax2.set_xlim([x.min(),x.max()])

    ax.legend()
    ax2.legend()

    plt.show()

