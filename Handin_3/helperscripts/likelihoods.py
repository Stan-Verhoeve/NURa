import numpy as np

##########################
## Gaussian likelihoods ##
##########################

def gaussian_logL(data, model, sigma, p):
    """
    Gaussian log-likelihood

    Parameters
    ----------
    data : ndarray
        Data to compare to. Expected shape (Npoints, Ndims).
        Expects data[:,-1] to be drawn from f(data[:,:-1], *p)
    model : callable
        Model to compare data to. Expected function call:
            model(x, *params)
    sigma : ndarray
        Standard deviation of the model. Expected to have same
        shape as data[:,-1]
    p : ndarray
        Model parameters

    Returns
    -------
    float
        log-likelihood assuming Gaussian errors
    """
    x, y = data[:,0], data[:,1]
    res = (y - model(x, *p)) / sigma
    return np.sum(res**2)

def gaussian_logL_gradient(data, model, sigma, derivatives, p):
    """
    Gaussian log-likelihood derivatives wrt model parameters

    Parameters
    ----------
    data : ndarray
        Data to compare to. Expected shape (Npoints, Ndims).
        Expects data[:,-1] to be drawn from f(data[:,:-1], *p)
    model : callable
        Model to compare data to. Expected function call:
            model(x, *params)
    sigma : ndarray
        Standard deviation of the model. Expected to have same
        shape as data[:,-1]
    derivatives : tuple
        Tuple of callables of same shape as p. Expects
        derivatives[i] to correspond to p[i].
    p : ndarray
        Model parameters

    Returns
    -------
    float
        log-likelihood gradient assuming Gaussian errors
    """
    x, y = data[:, 0], data[:, 1]
    f = model(x, *p)

    # Jacobian
    J = [df(x, *p) for df in derivatives]
    J = np.stack(J, axis=1)
    res = (y - f) / sigma**2

    return -2 * J.T @ res

############################
## Poissonian likelihoods ##
############################

def poissonian_logL(data, model, sigma, p):
    """
    Poissonian log-likelihood

    Parameters
    ----------
    data : ndarray
        Data to compare to. Expected shape (Npoints, Ndims).
        Expects data[:,-1] to be drawn from f(data[:,:-1], *p)
    model : callable
        Model to compare data to. Expected function call:
            model(x, *params)
    sigma : ndarray
        Standard deviation of the model. Expected to have same
        shape as data[:,-1]
    p : ndarray
        Model parameters

    Returns
    -------
    float
        log-likelihood assuming Poissonian errors
    """
    x, y = data[:, 0], data[:, 1]
    y_model = model(x, *p)
    return np.sum(y * np.log(y_model + 1e-10) - y_model)

def poissonian_logL_gradient(data, model, sigma, derivatives, p):
    """
    Poissonian log-likelihood derivatives wrt model parameters

    Parameters
    ----------
    data : ndarray
        Data to compare to. Expected shape (Npoints, Ndims).
        Expects data[:,-1] to be drawn from f(data[:,:-1], *p)
    model : callable
        Model to compare data to. Expected function call:
            model(x, *params)
    sigma : ndarray (NOT USED)
        Standard deviation of the model. Expected to have same
        shape as data[:,-1]
    derivatives : tuple
        Tuple of callables of same shape as p. Expects
        derivatives[i] to correspond to p[i].
    p : ndarray
        Model parameters

    Returns
    -------
    float
        log-likelihood gradient assuming Poissonian errors
    """
    x, y = data[:, 0], data[:, 1]
    f = model(x, *p)
    
    # Jacobian
    J = [df(x, *p) for df in derivatives]
    J = np.stack(J, axis=1)
    res = y / f - 1
    return J.T @ res

