import psutil
import time

from result import Ok, Err

def has_handle(fpath):
    """Check if a file has an open handle

    Positional Arguments:
    fpath -- a string path to the file

    Returns
    boolean -- True if a handle is found, False otherwise
    """
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False

def wait_for_no_handles(files, timeout=60):
    """Wait for a list of files to have no open handles

    This is because the screenshot app I'm using is slow to write the files. So wait until nothing has an
    open file handles on any of hte files, or we timeout.

    Positional Arguments:
    files -- a list of string file paths

    Keyword Arguments:
    timeout -- an integer with how many times to try and check. default is 60

    Returns:
    Result -- Ok if there are no open file handles, Err if we reached a timeout
    """
    passes = 0
    success_count = 0
    while passes < timeout and success_count < 3:
        time.sleep(1) # sleep first because we expect the first couple of passes to fail anyway
        handles = [has_handle(f) for f in files]
        if any(handles):
            passes += 1
            continue
        success_count += 1
        passes += 1
    # yeah, this is a bit wonky but I wanna try it anyway
    if success_count > 0:
        return Ok()

    return Err("In wait_for_handles, we hit the timeout while still finding open file handles")

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
