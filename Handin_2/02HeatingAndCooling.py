from numpy import log

# Constants
ALPHA_B = 2e-13
PSI = 0.929
TC = 1e4
Z = 0.015
Z2_inv = 1 / (Z**2)
KB = 1.38e-16
A = 5e-10
XI = 1e-15


def heating(T, nH):
    return ALPHA_B * nH * PSI * KB * TC + A * XI + 8.9e-30 * T


def heating_deriv(T, nH):
    return 8.9e-30


def cooling(T, nH):
    return (
        ALPHA_B
        * nH
        * KB
        * T
        * (0.684 - 0.0416 * log(T * Z2_inv * 1e-4) + 0.54 * (T * 1e-4) ** (0.37))
    )


def cooling_deriv(T, nH):
    return (
        ALPHA_B
        * nH
        * KB
        * (
            0.684
            - 0.0416 * (log(T * Z2_inv * 1e-4) + 1)
            + 0.54 * 1.37 * (T * 1e-4) ** 0.37
        )
    )


def equilibrium(T):
    return PSI * TC - (0.684 - 0.0416 * log(T * Z2_inv * 1e-4)) * T


def equilibrium2(T, nH):
    return heating(T, nH) - cooling(T, nH)


def equilibrium2_deriv(T, nH):
    return heating_deriv(T, nH) - cooling_deriv(T, nH)


def main():
    from helperscripts.root import false_position, newton_raphson

    # Initial bracket
    bracket = (1, 1e7)

    # Find roots
    # Use false_position to ensure new bracket does not go outside
    # original bracket. This is important to avoid invalid values
    # in the log
    root, aerr, rerr = false_position(
        equilibrium, bracket, atol=100, rtol=1e-5, max_iters=int(1e5)
    )

    print("Root found using false-position")
    print(f"    Equilibrium temperature: {root} K")
    print(f"    Absolute error: {aerr}")
    print(f"    Relative error: {rerr}\n")

    #########
    ## Q2b ##
    #########

    # Initial bracket
    bracket = (1, 1e15)
    x0 = 5e14

    for nH in [1e-4, 1, 1e4]:
        func = lambda T: equilibrium2(T, nH)
        func_deriv = lambda T: equilibrium2_deriv(T, nH)
        root, aerr, rerr = newton_raphson(
            func, func_deriv, x0, atol=1e-10, rtol=1e-10, max_iters=int(1e5)
        )

        print(f"Density nH = {nH}")
        print(f"    Equilibrium temperature: {root} K")
        print(f"    Absolute error: {aerr}")
        print(f"    Relative error: {rerr}\n")


if __name__ in ("__main__"):
    main()
