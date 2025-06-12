import numpy as np

def quasi_newton(
    f: callable,
    grad_f: callable,
    p0: np.ndarray,
    max_iters: int = 10,
    atol: float = 1e-2,
):
    p = np.array(p0)
    H = np.eye(len(p))
    theta_history = np.zeros((max_iters, len(p)))

    for i in range(max_iters):
        grad = grad_f(p)
        if np.linalg.norm(grad) < atol:
            break

        step = -H @ grad

        # Line search
        alpha = 1.0
        c = 1e-4
        rho = 0.9
        while f(p + alpha * step) > f(p) + c * alpha * (grad @ step):
            alpha *= rho

        delta = alpha * step
        p_new = p + delta
        grad_new = grad_f(p_new)
        d = grad_new - grad

        delta_d = np.dot(delta, d)
        Hd = H @ d

        if delta_d == 0 or np.dot(d, Hd) == 0:
            break

        rho_inv = delta_d
        rho_val = 1.0 / rho_inv

        u = delta * rho_val - Hd / np.dot(d, Hd)

        H += (
            rho_val * np.outer(delta, delta)
            - np.outer(Hd, Hd) / np.dot(d, Hd)
            + (np.dot(d, Hd) * np.outer(u, u))
        )

        p = p_new
        theta_history[i] = p

    return p, theta_history[:i]

