import torch
import gpytorch

def rsens2(model, X):
    likelihood = model.likelihood
    if (isinstance(model.likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood)):

        with torch.no_grad():
            pred_x = likelihood(model(X))

        if (len(list(X.size())) == 1):
            V = pred_x.variance
        elif (len(list(X.size())) == 2):
            V = pred_x.variance.view(-1,1,1)
        else:
            raise ValueError("X must be 1- or 2-dimensional.")

        # evaluate dE[f]/dx and dVar[f]/dx
        d2E_d2x, d2V_d2x = predictive_gradients(model, X, interaction = True)

        return torch.sqrt((d2E_d2x)**2/V + (d2V_d2x)**2*0.5/(V)**2)
    elif (isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood)):
        raise ValueError("R-sens2  is not yet implemented for classification")
    else:
        raise ValueError("Unknown likelihood. Supported are GaussianLikelihood")

def rsens(model, X):
    likelihood = model.likelihood
    # evaluate dE[f]/dx and dVar[f]/dx
    dE_dx, dV_dx = predictive_gradients(model, X)
    if (isinstance(model.likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood)):

        with torch.no_grad():
            pred_x = likelihood(model(X))

        if (len(list(X.size())) == 1):
            V = pred_x.variance
        elif (len(list(X.size())) == 2):
            V = pred_x.variance.view(-1,1)
        else:
            raise ValueError("X must be 1- or 2-dimensional.")

        return torch.sqrt((dE_dx)**2/V + (dV_dx)**2*0.5/(V)**2)
    elif (isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood)):
        pred_x = model(X)
        E = pred_x.mean
        V = pred_x.variance

        inv_var_plus_one = 1/(1+V)
        mean_var_ratio = E/torch.sqrt(1+V)

        std_normal = torch.distributions.normal.Normal(0, 1)
        prob_ratio = torch.exp(std_normal.log_prob(mean_var_ratio))
        cdf_ratio = std_normal.cdf(mean_var_ratio)

        dKL_dP = 1/cdf_ratio * 1/(1-cdf_ratio)
        dP_dx = prob_ratio * (torch.sqrt(inv_var_plus_one) * dE_dx - 0.5*inv_var_plus_one*mean_var_ratio* dV_dx)

        return torch.sqrt(dKL_dP * (dP_dx ** 2))
    else:
        raise ValueError("Unknown likelihood. Supported likelihoods are GaussianLikelihood and BernoulliLikelihood.")

def absderiv(model, X):
    likelihood = model.likelihood
    # evaluate dE[f]/dx and dVar[f]/dx
    dE_dx, dV_dx = predictive_gradients(model, X)
    if (isinstance(model.likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood)):
        return torch.abs(dE_dx)
    elif (isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood)):
        pred_x = model(X)
        E = pred_x.mean
        V = pred_x.variance

        inv_var_plus_one = 1/(1+V)
        mean_var_ratio = E/torch.sqrt(1+V)

        std_normal = torch.distributions.normal.Normal(0, 1)
        prob_ratio = torch.exp(std_normal.log_prob(mean_var_ratio))

        dP_dx = prob_ratio * (torch.sqrt(inv_var_plus_one) * dE_dx - 0.5*inv_var_plus_one*mean_var_ratio* dV_dx)

        return torch.abs(dP_dx)
    else:
        raise ValueError("Unknown likelihood. Supported likelihoods are GaussianLikelihood and BernoulliLikelihood.")


def predictive_gradients(model, X, interaction = False):

    # evaluate the model at the new points
    with gpytorch.settings.fast_pred_var():
        X.requires_grad_()
        pred_x = model(X)

    E = pred_x.mean
    V = pred_x.variance

    # calculate dE[f]/dx
    dE_dx = torch.autograd.grad(outputs=E, inputs=X, grad_outputs=torch.ones(X.shape[0]), retain_graph=True, create_graph = True)[0]
    #calculate dVar[f]/dx
    dV_dx = torch.autograd.grad(outputs=V, inputs=X, grad_outputs=torch.ones(X.shape[0]), retain_graph=True, create_graph = True)[0]

    if (interaction):
        N = X.size(0)
        if (len(list(X.size())) == 2):
            D = X.size(1)

        d2E_d2x = torch.zeros((N,D,D))
        d2V_d2x = torch.zeros((N,D,D))
        for d in range(1,D):
            d2E_d2x[:,:,d] = torch.autograd.grad(outputs=dE_dx[:,d], inputs=X, grad_outputs=torch.ones_like(E), retain_graph=True, create_graph = True)[0]
            d2E_d2x[:,torch.arange(D)[torch.arange(D) > (d-1)],d] = 0
            d2V_d2x[:,:,d] = torch.autograd.grad(outputs=dV_dx[:,d], inputs=X, grad_outputs=torch.ones_like(E), retain_graph=True, create_graph = True)[0]
            d2V_d2x[:,torch.arange(D)[torch.arange(D) > (d-1)],d] = 0
        # detach
        X.requires_grad_(False)
        d2E_d2x = d2E_d2x.detach()
        d2V_d2x = d2V_d2x.detach()

        return d2E_d2x, d2V_d2x
    else:
        # detach
        X.requires_grad_(False)
        dE_dx = dE_dx.detach()
        dV_dx = dV_dx.detach()
        return dE_dx, dV_dx
