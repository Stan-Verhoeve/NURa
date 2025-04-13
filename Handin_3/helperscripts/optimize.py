import numpy as np
from .sorting import merge_sort
from .linalg import solve_system

# Golden ratio to 12 decimals
PHI = 1.618_033_988_749


def find_bracket(
    func: callable, a: float, b: float, args: tuple = (), max_iters: int = 100
):
    """
    Finds a valid bracket (a,b,c) for function minimalization, where
    f(b) < f(a) and f(b) < f(c), and a < b < c

    Parameters
    ----------
    func : callable
        Function to find bracket for
    a : float
        Initial left side of bracket
    b : float
        Initial right side of bracket
    args : tuple
        Arguments to be passed to func
    max_iters : int, optional
        Maximum number of iterations to find bracket.
        The default is 100

    Returns
    -------
    bracket : tuple
        Initial bracket for function
    """
    # Get function values at initial two points
    fa = func(a, *args)
    fb = func(b, *args)

    # If not decreasing, switch labels
    if fa < fb:
        fa, fb = fb, fa
        a, b = b, a

    # Proposed third point and function value
    c = b + (b - a) * (2 - PHI)
    fc = func(c, *args)

    # Found a bracket
    if fc > fb:
        return merge_sort([a, b, c])

    # Keep fitting parabolas to find a bracket
    # TODO: come back to see if this can be optimized
    for _ in range(max_iters):
        # If not a bracket, fit parabola minimum
        num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
        denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)

        if denom == 0:
            # Fall back to step if div-by-zero
            d = c + (c - b) * (2 - PHI)
        else:
            d = b - 0.5 * (num / denom)

        fd = func(d, *args)

        if b < d < c:
            if fd < fc:
                return merge_sort([b, d, c])
            elif fd > fb:
                return merge_sort([a, b, d])
        else:
            if abs(d - b) > 100 * abs(c - b):
                d = c + (c - b) * (2 - PHI)

    raise RuntimeError("Maximum iterations reached without finding a bracket")


def golden_section(func, a, b, args=(), atol=1e-3, rtol=1e-3, max_iters=100):
    bracket = find_bracket(func, a, b, args, max_iters)

    # Set mask to zero if left is largest. That way, we can grab the
    # largest interval using bracket[largest_mask:1+largest_mask]
    # which will return (a,b) if left is largest, and (b,c) if right is largest
    largest_mask = abs(bracket[1] - bracket[0]) < abs(bracket[2] - bracket[1])

    for _ in range(max_iters):
        # If left interval, this reverses the order (a,b) to (b,a),
        # whereas if right interval, this keeps the order (b,c)
        # This guarantees x is always the other edge of the largest
        # interval
        b, x = bracket[largest_mask : largest_mask + 2][
            :: (-1) ** (largest_mask + 1)
        ]

        # Index of the point x
        idx = 2 * (1 - largest_mask)

        # Propose new point
        d = b + (x - b) * (2 - PHI)

        # Return if desired tolerance has been reached
        if abs(bracket[2] - bracket[0]) < atol:
            return d if func(d, *args) < func(b, *args) else b

        if func(d, *args) < func(b, *args):
            # if between a and b (left interval), largest_mask = 0
            # and we need c=b and b=d
            # so bracket[-1] == bracket[-2]  (equiv: bracket[2] == bracket[1])
            # and bracket[-2] == d           (equiv: bracket[1] == d

            # If between b and c (rigth interval), largest_mask = 1
            # and we need a=b and b=d
            # so bracket[0] == bracket[1]    (equiv: bracket[0] == bracket[1])
            # and bracket[1] == d            (equiv: bracket[1] == d
            bracket[idx] = bracket[1]
            bracket[1] = d

        else:
            # if between b and c (right interval), largest_mask = 1
            # and we need c=d --> bracket[2] = d

            # if between a and b (left interval), largest_mas = 0
            # and we need a=d --> bracket[0] = d
            bracket[idx] = d

            # New largest is one we did not tighten.
            # Only if func(d) >= func(b) do we switch
            # which interval we tightened
            largest_mask = not largest_mask

    raise RuntimeError("Maximum iterations reached without finding a minimum")


def levenberg_marquardt(
    data: np.ndarray,
    model: callable,
    sigma: callable,
    derivatives: tuple,
    logL: callable,
    dlogL_dp: callable,
    p0: np.ndarray,
    DoF: float = 1,
    step: float = 1e-3,
    weight: float = 10,
    max_iters: int = 100,
    atol: float = 0.01,
):
    """
    Levenberg-Marquardt routine to maximize a chi-squared problem.
    NOTE: ALWAYS assumes chi-squared / least squared. This method
          assumes a minimization of squared residuals

    Parameters
    ----------
    data : ndarray
        Array containing data of the problem. Assumes data[0] to be the input vector,
        and data[1] to be the output vector
    model : callable
        Model function to fit to. Expected to have the following function call:
            model(data, *params)
    sigma : callable
        Model variances. Expects same function call as model
    derivatives : tuple
        tuple of callables. Derivatives of the model to each of its parameters
        Expects derivatives[i] to correspond to p0[i]
    logL : callable
        log-likelihood function to maximize. Expects the following call structure:
            logL(data, model, sigma, params)
    dlogL_dp : callable
        Derivative of log-likelihood function to each of its parameters.
        Expectes the following call structure:
            dlogL_dp(data, model, sigma, params)
    p0 : array_like
        Initial guess for fit parameters. Expected to have same shape
        as derivatives. Expects p[i] to correspond to derivatives[i]
    step : float, optional
        Initial step lambda. The default is 1e-3
    weight : float, optional
        Weighing / damping of the steps. The default is 10
    max_iters : int, optional
        Maximum number of iterations before returning current best fit.
        The default is 100
    atol : float, optional
        Tolerance in logL improvement before returning parameters.
        The default is 0.01

    Returns
    -------
    p : array_like
        Best-fitting parameters. Has same shape as p0
    """
    # Current best guess
    p = np.array(p0)

    # Data
    x = data[0]
    y = data[1]

    # Pre-calculate
    weight_inv = 1 / weight
    DoF_inv = 1 / DoF
    sm = sigma(x, *p)

    # Previous logL to compare to
    logL_prev = logL(data, model, sm, p)

    for _ in range(max_iters):
        # Standard deviation of model under consideration
        sm = sigma(x, *p)
        sigma_inv = 1 / sm

        # Current function value
        f = model(x, *p)

        # Jacobian matrix
        J = [df(x, *p) * sigma_inv for df in derivatives]
        J = np.stack(J, axis=1)

        # Pseudo-hessian
        alpha = J.T @ J
        beta = -0.5 * dlogL_dp(data, model, sm, derivatives, p)

        # Step between steepest and Newton
        # Use diag(diag(alpha)) because diag(alpha) --> 1D array, diag(1D) --> square matrix with 1D on diagonal
        alpha_prime = alpha + step * np.diag(np.diag(alpha))

        # Solve for dp
        dp = solve_system(alpha_prime, beta)
        p_new = p + dp

        # New log-likelihood
        logL_new = logL(data, model, sm, p_new)
        logL_diff = logL_new - logL_prev

        # New parameters are worse, do not accept
        # logL_new > logL_prev --> logL_new - logL_prev > 0
        if logL_diff > 0:
            newstep = step * weight
            step = min(newstep, 1e6)
        else:
            # New parameters are better, accept
            p = p_new
            step *= weight_inv

            # Update old value
            logL_prev = logL_new

        # Return early if (almost) no improvement
        # Assumes atol to be in units of reduced chi-squared
        if abs(logL_diff) * DoF_inv < atol:
            return p

    print("Max iters reached")
    return p


def quasi_newton(
    data: np.ndarray,
    model: callable,
    sigma: callable,
    derivatives: tuple,
    logL: callable,
    dlogL_dp: callable,
    p0: np.ndarray,
    max_iters: int = 10,
    atol: float = 0.01,
):
    # Current parameter
    p = np.array(p0)
    x = data[0]
    y = data[1]

    I = np.eye(len(p))
    sm = sigma(x, *p)
    H = I.copy()

    for _ in range(max_iters):
        # Recalculate expected error
        # sm = sigma(x, *p)

        # Calculate gradient and step in direction
        gradient = dlogL_dp(data, model, sm, derivatives, p)
        if np.linalg.norm(gradient) < atol:
            break
        step = -1 * H @ gradient

        # New parameters (assumes lambda=1)
        p_new = p + step
        logL_new = logL(data, model, sm, p_new)

        # Line search
        alpha = 1.0
        c = 1e-4
        rho = 0.9

        # Check if model at new params is worse than expected based on gradient
        while logL(data, model, sm, p + alpha * step) > logL(
            data, model, sm, p
        ) + c * alpha * (gradient @ p):
            alpha *= rho

        # Calculate delta, and new gradient
        delta = alpha * step  # <-- Delta
        p_new = p + delta  # <-- xi+1
        grad_new = dlogL_dp(data, model, sm, derivatives, p_new)
        d = grad_new - gradient  # <-- d

        # Inner products
        delta_d = np.dot(delta, d)  # <-- delta dot d
        Hd = np.dot(H, d)  # <-- H dot d

        # Avoid div-by-zero
        if delta_d == 0.0:
            break

        # One over (delta dot d)
        rho_inv = delta_d
        rho_val = 1.0 / rho_inv  # <-- 1 / (delta dot d)

        # U from slides
        u = delta * rho_val - Hd / np.dot(d, Hd)

        # Update Hessian matrix (equation taken from slides)
        H += (
            (np.outer(delta, delta) * rho_val)
            - (np.outer(Hd, Hd) / np.dot(d, Hd))
            + (np.dot(d, Hd) * np.outer(u, u))
        )

        p = p_new
    return p
