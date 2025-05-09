import numpy as np
import matplotlib.pyplot as plt

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

def system_q1a(x, state):
    return -1 * state

def system_q1b(x, state):
    y, z = state
    
    return np.array([z, y])

def analytic_q1a(t):
    return 2 * np.exp(-t)

def analytic_q1b(t):
    return 8 * np.exp(-t) + 7 * np.exp(t)

def main():
    #########
    ## Q1a ##
    #########
    trange = (0, 20)
    init = [2.0]
    h = 0.5
    
    time, sols_euler = solve_ivp(system_q1a, trange, init, h, method="euler")
    time, sols_rk4 = solve_ivp(system_q1a, trange, init, h, method="rk4")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(time, analytic_q1a(time), c="gray", lw=5, alpha=0.3, label="analytic")
    ax.plot(time, sols_euler, c="r", label="Euler")
    ax.plot(time, sols_rk4, c="k", ls="--", label="RK4")
    
    ax.legend()
    ax.set(title=f"Solving dy/dt = -y, h={h}",
           xlabel="t",
           ylabel="y",
           yscale="log",
           )

    fig.savefig("figures/01_ODEs_Q1a.png", bbox_inches="tight", dpi=600)
    
    #########
    ## Q1b ##
    #########
    trange = (0, 10)
    init = [15, -1]
    h = 0.1

    time, sols_euler = solve_ivp(system_q1b, trange, init, h, method="euler")
    time, sols_rk4 = solve_ivp(system_q1b, trange, init, h, method="rk4")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(time, analytic_q1b(time), c="gray", lw=5, alpha=0.3, label="analytic")
    ax.plot(time, sols_euler[:,0], c="r", label="Euler")
    ax.plot(time, sols_rk4[:,0], c="k", ls="--", label="RK4")

    ax.legend()
    ax.set(title=rf"Solving $\frac{{d^2y}}{{dt^2}} = y$, h = {h}",
           xlabel="t",
           ylabel="y",
           yscale="log",
           )

    fig.savefig("figures/01_ODEs_Q1b.png", bbox_inches="tight", dpi=600)


if __name__ in ("__main__"):
    main()
