import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import my_functions as mf

# Scalar-vector
def test_scalar_vec_real_a_not_scalar():
    'fail if a is not a scalar'
    # 2d array
    a1 = np.ones((3,2))
    # list
    a2 = [7.]
    # tuple
    a3 = (4, 8.2)
    vector = np.arange(4)
    for ai in [a1, a2, a3]:
        with pytest.raises(AssertionError):
            mf.scalar_vec_real(ai, vector)

def test_scalar_vec_real_x_not_1darray():
    'fail if x is not a 1d array'
    a = 2
    # 2d array
    x1 = np.ones((3,2))
    # string
    x2 = 'not array'
    for xi in [x1, x2]:
        with pytest.raises(AssertionError):
            mf.scalar_vec_real(a, xi)


def test_scalar_vec_real_known_values():
    'check output produced by specific input'
    scalar = 1
    vector = np.linspace(23.1, 52, 10)
    reference_output = np.copy(vector)
    
    computed_output_dumb  = mf.scalar_vec_real(scalar, vector)
    computed_output_python = scalar*vector
    
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_python, decimal=10)
    aae(computed_output_dumb, computed_output_python, decimal=10)

def test_scalar_vec_complex_compare_numpy():
    'compare scalar_vec_complex with numpy'
    # set random generator
    rng = np.random.default_rng(763412)
    # use the random generator to create input parameters
    scalar = rng.random() + 1j*rng.random()
    vector = rng.random(13) + rng.random(13)*1j
    output = mf.scalar_vec_complex(scalar, vector, check_input=True)
    reference = scalar*vector
    aae(output, reference, decimal=10)