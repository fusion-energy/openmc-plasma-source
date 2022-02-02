"""properties

A collection of utility functions for implementing class properties
throughout openmc-plasma-source.
"""


def property_factory(
    property_name: str,
    condition=None,
    transform=None,
    condition_err_msg: str = "",
    transform_err_msg: str = "",
):
    """A generic function for creating class properties.

    This function allows the user to assert a condition that must be satisfied
    within the setter, along with the type the property should be set to. It
    does not impose any conditions on the getter.

    Parameters:

        property_name (str): Name of the class property. Must match exactly the
            name assigned, e.g. my_attr = property_factory( "my_attr", ... ).
        condition : Should either be None, meaning any input is accepted, or a
            function taking one argument and returning a bool. An example of a
            common condition may be the property must be non-negative.
        transform : Should either be None, meaning the value is not modified,
            or function that takes one argument. This is used to modify inputs
            to the property. If set to a type (e.g. int, float, str), transform
            will attempt to cast the input to that type.
        condition_err_msg (str): Message to print if 'condition' returns False.
        transform_err_msg (str): Message to print if casting to 'transform'
            fails.

    Returns:

        property : A property decorator. For details on properties, see
            https://realpython.com/python-property/
    """

    def getter(instance):
        """Retrieve the property from the given class instance."""
        return instance.__dict__[property_name]

    def setter(instance, value):
        """Assign 'value' to the property within the given class instance."""

        if condition is not None:
            try:
                valid = condition(value)
            except Exception as e:
                err_msg = (
                    f"Setter condition for property "
                    f"{instance.__class__.__name__}.{property_name} "
                    f"raised exception. See traceback for more info."
                )
                if condition_err_msg:
                    err_msg += f"\n{condition_err_msg}"
                raise ValueError(err_msg) from e

            if not valid:
                err_msg = (
                    f"Setter condition for property "
                    f"{instance.__class__.__name__}.{property_name} "
                    f"returned False."
                )
                if condition_err_msg:
                    err_msg += f"\n{condition_err_msg}"
                raise ValueError(err_msg)

        if transform is not None:
            try:
                value = transform(value)
            except Exception as e:
                err_msg = (
                    f"Setter transform for property "
                    f"{instance.__class__.__name__}.{property_name} "
                    f"raised exception. See traceback for more info."
                )
                if transform_err_msg:
                    err_msg += f"\n{transform_err_msg}"
                raise ValueError(err_msg) from e

        instance.__dict__[property_name] = value

    return property(getter, setter)


def positive_float(property_name: str, no_zero=False):
    """Creates property that must greater than or equal to 0"""
    return property_factory(
        property_name,
        condition=lambda x: x > 0 if no_zero else x >= 0,
        condition_err_msg="Must be greater than or equal to 0",
        transform=float,
        transform_err_msg="Must be convertable to float",
    )


def positive_int(property_name: str, no_zero=False):
    """Creates property that must be integer and greater than or equal to 0"""
    return property_factory(
        property_name,
        condition=lambda x: x > 0 if no_zero else x >= 0,
        condition_err_msg="Must be greater than or equal to 0",
        transform=int,
        transform_err_msg="Must be convertable to int",
    )


def in_range(property_name: str, bounds):
    """Creates property that must be between bounds[0] and bounds[1]."""
    return property_factory(
        property_name,
        condition=lambda x: x >= bounds[0] and x <= bounds[1],
        condition_err_msg=(
            f"Must be within the range [{bounds[0]},{bounds[1]}] (inclusive)"
        ),
    )