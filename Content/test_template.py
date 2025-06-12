import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import template as tmp

# Scalar-vector test
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
            tmp.scalar_vec_real(ai, vector)

def test_scalar_vec_real_x_not_1darray():
    'fail if x is not a 1d array'
    a = 2
    # 2d array
    x1 = np.ones((3,2))
    # string
    x2 = 'not array'
    for xi in [x1, x2]:
        with pytest.raises(AssertionError):
            tmp.scalar_vec_real(a, xi)


def test_scalar_vec_real_known_values():
    'check output produced by specific input'
    scalar = 1
    vector = np.linspace(23.1, 52, 10)
    reference_output = np.copy(vector)
    
    computed_output_dumb  = tmp.scalar_vec_real(scalar, vector)
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
    
    output = temp.scalar_vec_complex(scalar, vector, check_input=True)
    reference = scalar*vector
    
    aae(output, reference, decimal=10)

# Dot test
def test_dot_real_not_1D_arrays():
    'fail due to input that is not 1D array'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        temp.dot_real(vector_1, vector_2)


def test_dot_real_different_sizes():
    'fail due to inputs having different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        temp.dot_real(vector_1, vector_2)


def test_dot_real_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output = temp.dot_real(vector_1, vector_2)
    aae(reference_output, computed_output, decimal=10)


def test_dot_real_compare_numpy_dot():
    'compare with numpy.dot'
    # set random generator
    rng = np.random.default_rng(12765)
    # use the random generator to create input parameters
    vector_1 = rng.random(13)
    vector_2 = rng.random(13)
    reference_output_numpy = np.dot(vector_1, vector_2)
    computed_output = temp.dot_real(vector_1, vector_2)
    aae(reference_output_numpy, computed_output, decimal=10)


def test_dot_real_commutativity():
    'verify commutativity'
    # set random generator
    rng = np.random.default_rng(555543127)
    # use the random generator to create input parameters
    a = rng.random(15)
    b = rng.random(15)
    # a dot b = b dot a
    output_ab = temp.dot_real(a, b)
    output_ba = temp.dot_real(b, a)
    aae(output_ab, output_ba, decimal=10)


def test_dot_real_distributivity():
    'verify distributivity over sum'
    # set random generator
    rng = np.random.default_rng(555543127)
    # use the random generator to create input parameters
    a = rng.random(15)
    b = rng.random(15)
    c = rng.random(15)
    # a dot (b + c) = (a dot b) + (a dot c)
    output_a_bc = temp.dot_real(a, b + c)
    output_ab_ac = temp.dot_real(a, b) + temp.dot_real(a, c)
    aae(output_a_bc, output_ab_ac, decimal=10)


def test_dot_real_scalar_multiplication():
    'verify scalar multiplication property'
    # set random generator
    rng = np.random.default_rng(333543127)
    # use the random generator to create input parameters
    a = rng.random(15)
    b = rng.random(15)
    c1 = 5.6
    c2 = 9.1
    # (c1 a) dot (c2 b) = c1c2 (a dot b)
    output_c1a_c2b = temp.dot_real(c1*a, c2*b)
    output_c1c2_ab = c1*c2*temp.dot_real(a, b)
    aae(output_c1a_c2b, output_c1c2_ab, decimal=10)


def test_dot_complex_compare_numpy_dot():
    'compare dot_complex, numpy and numba with numpy.dot'
    # set random generator
    rng = np.random.default_rng(1111763412)
    # use the random generator to create input parameters
    vector_1 = rng.random(13) + 1j*rng.random(13)
    vector_2 = rng.random(13) + 1j*rng.random(13)
    output = temp.dot_complex(vector_1, vector_2)
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output, output_numpy_dot, decimal=10)
