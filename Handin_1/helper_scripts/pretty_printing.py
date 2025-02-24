from numpy import asarray, ndarray, array2string, pad, nan, shape, float64, isnan, ceil

def pretty_print_array(array: list | ndarray, formatter: str =".2e", ncols: int=None) -> None:
    """
    Prints a list or array with some formatting options.

    Parameters
    ----------
    array : list | ndarray
        Array to pretty print
    formatter : str
        Which f-string format to use when printing.
        The default is ".2e"
    ncols : int
        Number of columns to display array in. If 'None',
        defaults to number of columns (2D) or len (1D).
        The default is None
    """
    # Check of cols are given
    if not ncols:
        array_shape = shape(array)
        if len(array_shape) > 1:
            ncols = array_shape[1]
        else:
            ncols = array_shape[0]

    # Total length of (flattened) array
    N = len(asarray(array, dtype=float64).flatten())
    nrows = int(ceil(N / ncols))
    
    # Pad array with NaN at the end (if needed)
    padded = pad(array, (0, nrows * ncols - N), constant_values = nan)
    padded = padded.reshape(nrows,ncols)
    
    # Format only where not nan
    format_func = lambda x: f"{x:{formatter}}" if not isnan(x) else "     "
    
    # Print array
    print(array2string(padded, formatter={"float_kind":format_func}, separator="   "))

    return

