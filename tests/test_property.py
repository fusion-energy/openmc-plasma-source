"""Tests if utility functions in properties.py function correctly."""

from openmc_plasma_source.properties import (
    property_factory,
    positive_float,
    positive_int,
    in_range,
)

import pytest

@pytest.fixture
def property_class():
    class MyClass:
        ends_in_dot = property_factory(
            "ends_in_dot",
            transform=lambda z: z + ".",
            transform_err_msg="Couldn't add to end of string",
        )

        equal_to_5 = property_factory(
            "equal_to_5",
            condition=lambda z: z == 5,
            condition_err_msg="Should be equal to 5",
        )

        less_than_5 = property_factory(
            "less_than_5",
            condition=lambda z: z < 5,
            condition_err_msg="Should be less than 5",
        )

        pos_float = positive_float("pos_float")
        pos_int = positive_int("pos_int")
        ranged = in_range("ranged",(-1.0,1.0))

        def __init__(self):
            self.ends_in_dot = "string"
            self.equal_to_5 = 5
            self.less_than_5 = 3
            self.pos_float = 1.0
            self.pos_int = 5
            self.ranged = 0.0

    return MyClass()

def test_property_factory_transform(property_class):
    # MyClass.ends_in_dot should always have '.' in the last place
    assert property_class.ends_in_dot[-1] == "."
    # Test reassigning repeats the transform
    property_class.ends_in_dot = "hello world"
    assert property_class.ends_in_dot == "hello world."
    # Test that it can happen recusively
    property_class.ends_in_dot = property_class.ends_in_dot
    assert property_class.ends_in_dot == "hello world.."
    # Test that it raises ValueError when passed a non string
    with pytest.raises(ValueError) as execinfo:
        property_class.ends_in_dot = 17
    assert "raise" in str(execinfo.value)
    # Ensure that a failed set doesn't modify
    assert property_class.ends_in_dot == "hello world.."

def test_property_factory_condition(property_class):
    # MyClass.equal_to_5 should fail when given anything except 5
    # Note that it should not raise an exception in the vast
    # majority of cases.
    assert property_class.equal_to_5 == 5
    assert isinstance(property_class.equal_to_5, int)
    # Test it works for floats
    property_class.equal_to_5 = 5.0
    assert property_class.equal_to_5 == 5.0
    assert isinstance(property_class.equal_to_5, float)
    # Test it returns false when given something else
    with pytest.raises(ValueError) as execinfo:
        property_class.equal_to_5 = 0
    assert "False" in str(execinfo.value)
    # Ensure that a failed set didn't change the result
    assert property_class.equal_to_5 == 5.0
    # MyClass.less_than_5 should instead raise an exception
    # if given a string. First test that it works:
    property_class.less_than_5 = 3
    assert property_class.less_than_5 == 3
    # Now try giving it a string
    with pytest.raises(ValueError) as execinfo:
        property_class.less_than_5 = "hello world."
    assert "raise" in str(execinfo.value)

def test_positive_float(property_class):
    # Ensure that the user can assign to a positive_float
    property_class.pos_float = 7.2
    assert property_class.pos_float == 7.2
    # Zero should be allowed
    property_class.pos_float = 0.0
    assert property_class.pos_float == 0.0
    # Ensure that the user cannot set a positive_float property to a negative
    with pytest.raises(ValueError) as execinfo:
        property_class.pos_float = -1.0
    assert "False" in str(execinfo.value)
    # Ensure the value is unchanged by a failed assignment
    assert property_class.pos_float == 0.0
    # Ensure that the user cannot set a non-floatable
    with pytest.raises(ValueError) as execinfo:
        property_class.pos_float = "hello world"
    assert "raise" in str(execinfo.value)
    # Ensure the value is unchanged by a failed assignment
    assert property_class.pos_float == 0.0

def test_positive_int(property_class):
    # Ensure that the user can assign to a positive_int
    property_class.pos_int = 7
    assert property_class.pos_int == 7
    # Zero should be allowed
    property_class.pos_int = 0
    assert property_class.pos_int == 0
    # floats should be cast to ints correctly
    property_class.pos_int = 5.2
    assert property_class.pos_int == 5
    # Ensure that the user cannot set to something negative
    with pytest.raises(ValueError) as execinfo:
        property_class.pos_int = -3
    assert "False" in str(execinfo.value)
    # Ensure the value is unchanged by a failed assignment
    assert property_class.pos_int == 5
    # Ensure that the user cannot set to something that can't be cast to int
    with pytest.raises(ValueError) as execinfo:
        property_class.pos_int = "hello world"
    assert "raise" in str(execinfo.value)
    # Ensure the value is unchanged by a failed assignment
    assert property_class.pos_int == 5

def test_in_range(property_class):
    # Ensure that the user can assign to a in_range
    property_class.ranged = 0.5
    assert property_class.ranged == 0.5
    # The edge values should be allowed
    property_class.ranged = 1.0
    assert property_class.ranged == 1.0
    property_class.ranged = -1.0
    assert property_class.ranged == -1.0
    # No type restrictions, should be able to set to int
    property_class.ranged = 1
    assert property_class.ranged == 1
    assert isinstance( property_class.ranged, int)
    # Ensure that the user cannot set to something larger than upper bound
    with pytest.raises(ValueError) as execinfo:
        property_class.ranged = 1.5
    assert "False" in str(execinfo.value)
    # Ensure that the user cannot set to something smaller than lower bound
    with pytest.raises(ValueError) as execinfo:
        property_class.ranged = -1.1
    assert "False" in str(execinfo.value)
    # Ensure the value is unchanged by a failed assignment
    assert property_class.ranged == 1
    # Ensure that the user cannot set to something that can't be compared
    with pytest.raises(ValueError) as execinfo:
        property_class.ranged = "hello world"
    assert "raise" in str(execinfo.value)
    # Ensure the value is unchanged by a failed assignment
    assert property_class.ranged == 1
