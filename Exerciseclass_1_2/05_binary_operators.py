def divisible_by_2m(num, m, use_shift = True):
    """
    Check if a number is divisble by a power of 2.

    Parameters
    ----------
    num: int
        Numerator
    m: int
        Power of 2, i.e. m=2 checks for 2**2
    use_shift: bool, optional
        Whether to use bit-shifting logic.
        The default is True
    
    returns
    -------
    bool
        True if divisible by power of 2, false otherwise
    """
    # Binary mask consisting of `m` ones
    if use_shift:
        bin_mask = (1 << m) - 1
    else:
        bin_mask = 2**m - 1

    # If the and-operator is equal to zero, the
    # number is divisible by 2**m, by virtue of
    # the last m digits being zero
    divisible = (bin_mask & num) == 0
    return divisible

def main():
    user_input = input("Provide integer number and power, separated by a comma, e.g. '12, 2'\n").split(",")
    num = int(user_input[0])
    power = int(user_input[1])

    divisible = divisible_by_2m(num, power, use_shift=False)
    print(f"{num} divisible by 2**{power}: {divisible} (no bit shifting)")

    divisible = divisible_by_2m(num, power)
    print(f"{num} divisible by 2**{power}: {divisible} (bit shifting)")

if __name__ in ("__main__"):
    main()
