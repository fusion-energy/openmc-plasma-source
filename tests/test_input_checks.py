"""Tests if utility functions in utils.py function correctly."""

from openmc_plasma_source.utils import (
    ensure_positive_float,
    ensure_positive_int,
    ensure_in_range,
)

import pytest


def test_positive_float():
    # Ensure that positive floats pass
    assert ensure_positive_float(7.2) == 7.2
    # Ensure negative floats don't
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_float(-6.2)
    assert "greater than or equal to 0" in str(excinfo.value)
    # Zero should be allowed
    assert ensure_positive_float(0) == 0.0
    # ...unless setting no_zero to true
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_float(0, no_zero=True)
    assert "greater than 0" in str(excinfo.value)
    # Exceptions should be raised if fed something not float-able
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_float("hello world")
    assert "convert" in str(excinfo.value)
    # Variable name should be given in error messages if provided
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_float(-5, var_name="Jerry")
    assert "Jerry" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_float("hello world", var_name="Fred")
    assert "Fred" in str(excinfo.value)


def test_positive_int():
    # Ensure that positive ints pass
    assert ensure_positive_int(7) == 7
    # Ensure negative ints don't
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_int(-6)
    assert "greater than or equal to 0" in str(excinfo.value)
    # Zero should be allowed
    assert ensure_positive_int(0) == 0
    # ...unless setting no_zero to true
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_int(0, no_zero=True)
    assert "greater than 0" in str(excinfo.value)
    # Exceptions should be raised if fed something not int-able
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_int("hello world")
    assert "convert" in str(excinfo.value)
    # Variable name should be given in error messages if provided
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_int(-5, var_name="Jerry")
    assert "Jerry" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_int("hello world", var_name="Fred")
    assert "Fred" in str(excinfo.value)
    # Rounding is optional
    assert ensure_positive_int(5.2, no_rounding=False) == 5
    # ... and should fail by default
    with pytest.raises(ValueError) as excinfo:
        ensure_positive_int(5.2)
    assert "round" in str(excinfo.value)
    # Rounding from something like 2.0 or 10.0 should work
    assert ensure_positive_int(10.0) == 10


def test_in_range():
    # Ensure that something clearly in range counts
    assert ensure_in_range(0, bounds=(-1, 1)) == 0
    # Ensure that the edges count
    assert ensure_in_range(-1, bounds=(-1, 1)) == -1
    assert ensure_in_range(1, bounds=(-1, 1)) == 1
    # ... and that they don't when disabled
    with pytest.raises(ValueError) as excinfo:
        ensure_in_range(-1, bounds=(-1, 1), inclusive=(False, True))
    assert "range" in str(excinfo.value)
    assert "(" in str(excinfo.value)
    assert "]" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ensure_in_range(1, bounds=(-1, 1), inclusive=(True, False))
    assert "range" in str(excinfo.value)
    assert ")" in str(excinfo.value)
    assert "[" in str(excinfo.value)
    # Exceptions should be raised if fed something not float-able
    with pytest.raises(ValueError) as excinfo:
        ensure_in_range("hello world", bounds=(1, 2))
    assert "convert" in str(excinfo.value)
    # And also if bounds is broken in some way
    # Variable name should be given in error messages if provided
    fail_states = [5, (2, 1), (0, 5, 10), (0, "hello world")]
    for fail_state in fail_states:
        with pytest.raises(ValueError) as excinfo:
            ensure_in_range(1.5, bounds=fail_state)
        assert "bounds" in str(excinfo.value)
    # Variable names should be in error messages if provided
    with pytest.raises(ValueError) as excinfo:
        ensure_in_range(2, bounds=(-1, 1), var_name="Jerry")
    assert "Jerry" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ensure_in_range("hello world", bounds=(-1, 1), var_name="Fred")
    assert "Fred" in str(excinfo.value)
    # ... but not if the bounds are broken
    with pytest.raises(ValueError) as excinfo:
        ensure_in_range(0, bounds=(-1, 1, 2), var_name="Amy")
    assert "Amy" not in str(excinfo.value)
