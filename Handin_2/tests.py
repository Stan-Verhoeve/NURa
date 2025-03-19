def test_integration():
    """
    Tests if the integration scheme works
    as intended
    """
    from numpy import isclose
    from helperscripts.integrate import romberg
    # Functions to integrate
    known_closed = lambda x: -x**2
    known_open = lambda x: x ** (-0.5)

    bounds = (0,1)
    closed_result = romberg(known_closed, bounds, m=15) # Analytic result: -1/3
    open_result = romberg(known_open, bounds, m=15) # Analytic result: 2

    # Show that setting lower bound small instead of 0 results in diverging integral
    wrong_open_result = romberg(known_open, (1e-8, 1), m=10) # Analytic resutl: 2

    print("Closed function x^2")
    print(f"Analytic: -1/3. Numerical: {closed_result}")
    print(f"Close: {isclose(-1/3, closed_result)}")
    print()
    print("Half-open function 1/sqrt(x), evaluated on (0, 1) with mid-point rule")
    print(f"Analytic: 2. Numerical: {open_result}")
    print(f"Close: {isclose(2, open_result, atol=1e-3)}")
    print()
    print("Half-open function 1/sqrt(x), evaluated on (1e-8, 1) with trapezoid")
    print(f"Analytic: 2. Numerical: {wrong_open_result}")
    print(f"Close: {isclose(2, wrong_open_result, atol=1e-3)}")

def test_random_generator():
    """
    Test if the random generation works
    as intended
    """
    from helperscripts.random import Random, pearson
    # Get uniformly distributed points
    generator = Random()
    uniform = generator.uniform(size=10_000)

    # Average and std of generated points
    avg = uniform.mean() # Expected: 0.5
    var = uniform.var() # Expected: 1/12
    
    # Correlation between successive numbers
    x = uniform[1:]
    y = uniform[:-1]
    corr = pearson(x, y)

    print(f"Average of generated points: {avg}. Expected: 0.5")
    print(f"Variance of generated points: {var}. Expected: 1/12 = 0.08333...")
    print(f"Correlation between successive numbers: {corr}")

def test_sorting():
    """
    Test if sorting works as intended
    """
    from helperscripts.sorting import merge_sort, is_sorted
    from helperscripts.random import Random
    generator = Random()
    uniform = generator.uniform(size=50)
    
    print(f"Before sorting, is_sorted: {is_sorted(uniform)}")
    sorted_array = merge_sort(uniform)
    print(f"After sorting, is_sorted: {is_sorted(sorted_array)}")

if __name__ in ("__main__"):
    print("=======================")
    print("Now testing integration")
    print("=======================")
    test_integration()

    print("\n============================")
    print("Now testing random generator")
    print("============================")
    test_random_generator()
    
    print("\n===================")
    print("Now testing sorting")
    print("===================")
    test_sorting()
