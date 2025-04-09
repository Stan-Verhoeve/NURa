from .sorting import merge_sort

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


def golden_section_gpt(func, a, b, args=(), atol=1e-3, rtol=1e-3, max_iters=100):
    bracket = find_bracket(func, a, b, args)
    a, b, c = bracket

    for _ in range(max_iters):
        # Decide which interval is bigger
        if abs(c - b) > abs(b - a):
            x = c
            d = b + (x - b) * (2 - PHI)
        else:
            x = a
            d = b + (x - b) * (2 - PHI)

        if abs(c - a) < atol:
            return d if func(d) < func(b) else b

        if func(d) < func(b):
            if x == c:
                a, b = b, d
            else:
                c, b = b, d
        else:
            if x == c:
                c = d
            else:
                a = d
