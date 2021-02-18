from result import Ok, Err

def railroad(*args, init_arg=None):
    """Run a series of functions that return Result on a given input, returning the final Result or first Err

    Positional Arguments:
    args -- an ordered list of single positional argument functions

    Keyword Arguments:
    init_arg -- the input for the first function

    Returns:
    Result -- will return the final Ok, or the first Err

    Exceptions:
    ValueError -- will be raised if args is empty or init_arg is None
    """
    if len(args) < 1:
        raise ValueError('At least one function must be specified in railroad')
    if not all([callable(f) for f in args]):
        raise ValueError('All positional arguments must be functions in railroad')
    if init_arg is None:
        raise ValueError('You must include an initial argument init_arg in railroad')
    r = args[0](init_arg)
    if r.is_err():
        return r

    for f in args[1:]:
        r = f(r.unwrap())
        if r.is_err():
            return r

    return r
