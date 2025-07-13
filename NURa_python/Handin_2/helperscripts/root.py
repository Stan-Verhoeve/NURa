from numpy import inf


def secant(
    func: callable,
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
) -> float:
    """
    Find a root of a function using secant method

    Parameters
    ----------
    func : callable
        Function to find root of
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        Teh default is 100

    Returns
    -------
    root : float
        Approximate root
    """
    # Extract bracket
    x0, x1 = bracket
    for i in range(max_iters):
        fx0 = func(x0)
        fx1 = func(x1)

        # New best guess
        xnew = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Convergence criteria
        aerr = abs(xnew - x1)
        if (aerr < atol) & (aerr < abs(x1 * rtol)):
            print(f"Best estimate found after {i+1} iterations")
            return xnew, aerr, abs(aerr / x1)

        x0, x1 = x1, xnew

    raise ValueError("Maximum iterations reached without converging to root")


def false_position(
    func: callable,
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
) -> float:
    """
    Find a root of a function using secant method

    Parameters
    ----------
    func : callable
        Function to find root of
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        Teh default is 100

    Returns
    -------
    root : float
        Approximate root
    """

    # Extract bracket and first evaluation
    a, b = bracket
    fa = func(a)
    fb = func(b)
    x = inf

    if fa * fb > 0:
        raise ValueError("Function values at initial points must have opposite sign")

    for i in range(max_iters):
        # New guess
        xnew = b - fb * (b - a) / (fb - fa)

        if func(a) * func(xnew) < 0:
            b = xnew
        elif func(xnew) * func(b) < 0:
            a = xnew
        else:
            raise ValueError("No new bracket was found")

        # Convergence criteria
        aerr = abs(xnew - x)
        if (aerr < atol) & (aerr < abs(rtol * x)):
            print(f"Best estimate found after {i} iterations")
            return xnew, aerr, abs(aerr / x)

        x = xnew

    raise ValueError("Maximum iterations reached without converging to root")


def newton_raphson(
    func: callable,
    dfunc: callable,
    x0: float,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
) -> float:
    """
    Find a root of a functio using Newton-Raphson method

    Parameters
    ----------
    func : callable
        Function to find root of
    dfunc : callable
        Derivative of function to find root of
    x0 : float
        Initial guess
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    max_iters : int, optional
        Maximum number of iterations.
        The default is 100.

    Returns
    -------
    root : float
        Approximate root of the function
    """

    # Initial guess
    x = x0

    # Iterate to update guess
    for i in range(max_iters):
        fx = func(x)
        dfdx = dfunc(x)

        # Avoid division by zero
        if dfdx == 0:
            raise ValueError(
                "Zero derivative encountered. A different initial guess is recommended"
            )

        # New best guess
        xnew = x - fx / dfdx

        # Convergence criteria
        aerr = abs(xnew - x)
        if (aerr < atol) & (aerr < abs(x * rtol)):
            print(f"Best estimate found after {i+1} iterations")
            return xnew, aerr, abs(aerr / x)

        # Update best guess
        x = xnew

    raise ValueError("Maximum iterations reached without converging to root")
