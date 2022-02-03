"""utils

Collects various functions for use throughout openmc-plasma-source.
As the project grows, consider moving these to more specialised
files."""


def ensure_positive_float(
    x,
    no_zero: bool = False,
    var_name: str = "",
    err_msg: str = "",
):
    """Test whether a given input is convertable to a positive float

    Parameters:

        x : Input of any type
        no_zero (bool): If true, excludes 0.0 as a positive float.
        var_name (str): Name of the variable to print in error messages.
        err_msg (str): Custom error message, overrides default.

    Returns:

        result (bool): True if convertable to positive float, False otherwise.
    """
    try:
        x = float(x)
    except ValueError as e:
        if err_msg:
            raise ValueError(err_msg)
        if var_name:
            raise ValueError(f"Could not convert {var_name}={x} to float.")
        raise ValueError(f"Could not convert {x} to float.")
    is_negative = (x <= 0) if no_zero else (x < 0)
    if is_negative:
        if err_msg:
            raise ValueError(err_msg)
        raise ValueError(
            f"Input {var_name} must be greater than "
            f"{'' if no_zero else 'or equal to '}0. "
            f"Input given was {x}."
        )
    return x


def ensure_positive_int(
    x,
    no_zero: bool = False,
    no_rounding: bool = True,
    var_name: str = "",
    err_msg: str = "",
):
    """Test whether a given input is convertable to a positive int

    Parameters:

        x : Input of any type
        no_zero (bool): If true, excludes 0.0 as a positive int.
        no_rounding (bool): If True, inputs such as 3.3 or 5.2 are not
            allowed. Inputs such as 2.0 and 10.0 are always allowed.
        var_name (str): Name of the variable to print in error messages.
        err_msg (str): Custom error message, overrides default.

    Returns:

        result (bool): True if convertable to positive int, False otherwise.
    """
    try:
        val = int(x)
    except ValueError as e:
        if err_msg:
            raise ValueError(err_msg)
        if var_name:
            raise ValueError(f"Could not convert {var_name}={x} to int.")
        raise ValueError(f"Could not convert {x} to int.")
    is_negative = x <= 0 if no_zero else x < 0
    if is_negative:
        if err_msg:
            raise ValueError(err_msg)
        raise ValueError(
            f"Input {var_name} must be greater than "
            f"{'' if no_zero else 'or equal to '}0. "
            f"Input given was {x}."
        )
    if x - val != 0 and no_rounding:
        if var_name:
            raise ValueError(
                f"Input {var_name}={x} could not convert to int without rounding."
            )
        raise ValueError(f"Input {x} could not convert to int without rounding.")
    return val


def ensure_in_range(
    x,
    bounds,
    inclusive=(True, True),
    var_name: str = "",
    err_msg: str = "",
):
    """Test whether a given input is within given bounds

    Parameters:

        x : Input of any type
        bounds : Must be subscriptable and of length 2. bounds[0] should be the
            lower bound of the range, while bounds[1] should be the upper bound
            of the range.
        inclusive : Must be subscriptable and of length 2. If inclusive[0]
            evaluates to True, the lower bound is inclusive. If inclusive[1]
            evaluates to True, the upper bound is inclusive. Othewise, in
            either case, the bounds are exclusive.
        var_name (str): Name of the variable to print in error messages.
        err_msd (str): Custom error message, overrides default.

    Returns:

        result (bool): True if within range, false otherwise.
    """
    try:
        x = float(x)
    except ValueError as e:
        if err_msg:
            raise ValueError(err_msg)
        if var_name:
            raise ValueError(f"Could not convert {var_name}={x} to float.")
        raise ValueError(f"Could not convert {x} to float.")
    try:
        float(bounds[0])
        float(bounds[1])
        if len(bounds) != 2:
            raise ValueError()
        if bounds[0] > bounds[1]:
            raise ValueError()
    except (TypeError, ValueError) as e:
        raise ValueError(
            "bounds should be subscriptable and of length 2. "
            "bounds[0] should be the lower bound, while bounds[1] "
            "should be the upper bound."
        )
    is_in_range = (x >= bounds[0] if inclusive[0] else x > bounds[0]) and (
        x <= bounds[1] if inclusive[1] else x < bounds[1]
    )
    if not is_in_range:
        if err_msg:
            raise ValueError(err_msg)
        raise ValueError(
            f"Input {var_name} must be within range "
            f"{'[' if inclusive[0] else '('}{bounds[0]},{bounds[1]} "
            f"{']' if inclusive[1] else ')'}. "
            f"Input given was {x}"
        )
    return x
