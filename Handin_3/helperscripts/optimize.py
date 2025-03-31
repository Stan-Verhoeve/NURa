# Golden ratio to 12 decimals
PHI = 1.618_033_988_749

def find_bracket(func, a, b, args=(), max_iters=100):
    """
    Finds a valid bracket (a,b,c) for function minimalization, where
    f(b) < f(a) and f(b) < f(c), and a < b < c

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
    fc = func(c)
    
    # Found a bracket
    if fc > fb:
        return (a, b, c)
    
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
                return (b, d, c)
            elif fd > fb:
                return (a, b, d)
        else:
            if abs(d - b) > 100 * abs(c - b):
                d = c + (c - b) * (2 - PHI)

    raise RuntimeError("Maximum iterations reached without finding a bracket")

def minimize(func, a, b, args=(), atol=1e-3, rtol=1e-3):
    bracket = find_bracket(func, a, b, args)
