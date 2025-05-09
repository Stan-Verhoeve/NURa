import numpy as np

def rk4_single(func, t, state, step):
    k1 = step * func(t, state)
    k2 = step * func(t + 0.5 * step, state + 0.5 * k1)
    k3 = step * func(t + 0.5 * step, state + 0.5 * k2)
    k4 = step * func(t + step, state + k3)

    return state + (k1 + 2 * (k2 + k3) + k4) / 6

def euler_single(func, t, state, step):
    return state + step * func(t, state)

def solve_ivp(func, t_span, y0, step, method="rk4"):
    half_step = 0.5 * step
    one_sixth = 1/6
    tvals = np.arange(*t_span, step)
    sols = np.zeros((len(tvals), len(y0)))
    sols[0] = y0

    if method == "rk4":
        get_next = rk4_single
    elif method == "euler":
        get_next = euler_single
    else:
        raise ValueError(f"Method `{method}` not supported")

    for i in range(1, len(tvals)):
        t = tvals[i-1]
        state = sols[i-1]

        sols[i] = get_next(func, t, state, step)

    return tvals, sols
