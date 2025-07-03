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
    
    output = tmp.scalar_vec_complex(scalar, vector, check_input=True)
    reference = scalar*vector
    
    aae(output, reference, decimal=10)

# Dot test
def test_dot_real_not_1D_arrays():
    'fail due to input that is not 1D array'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        tmp.dot_real(vector_1, vector_2)


def test_dot_real_different_sizes():
    'fail due to inputs having different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        tmp.dot_real(vector_1, vector_2)


def test_dot_real_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output = tmp.dot_real(vector_1, vector_2)
    aae(reference_output, computed_output, decimal=10)


def test_dot_real_compare_numpy_dot():
    'compare with numpy.dot'
    # set random generator
    rng = np.random.default_rng(12765)
    # use the random generator to create input parameters
    vector_1 = rng.random(13)
    vector_2 = rng.random(13)
    reference_output_numpy = np.dot(vector_1, vector_2)
    computed_output = tmp.dot_real(vector_1, vector_2)
    aae(reference_output_numpy, computed_output, decimal=10)


def test_dot_real_commutativity():
    'verify commutativity'
    # set random generator
    rng = np.random.default_rng(555543127)
    # use the random generator to create input parameters
    a = rng.random(15)
    b = rng.random(15)
    # a dot b = b dot a
    output_ab = tmp.dot_real(a, b)
    output_ba = tmp.dot_real(b, a)
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
    output_a_bc = tmp.dot_real(a, b + c)
    output_ab_ac = tmp.dot_real(a, b) + temp.dot_real(a, c)
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
    output_c1a_c2b = tmp.dot_real(c1*a, c2*b)
    output_c1c2_ab = c1*c2*tmp.dot_real(a, b)
    aae(output_c1a_c2b, output_c1c2_ab, decimal=10)


def test_dot_complex_compare_numpy_dot():
    'compare dot_complex, numpy and numba with numpy.dot'
    # set random generator
    rng = np.random.default_rng(1111763412)
    # use the random generator to create input parameters
    vector_1 = rng.random(13) + 1j*rng.random(13)
    vector_2 = rng.random(13) + 1j*rng.random(13)
    output = tmp.dot_complex(vector_1, vector_2)
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output, output_numpy_dot, decimal=10)

# Outer tests
def test_outer_real_input_not_vector():
    'fail with non-vector inputs'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        tmp.outer_real_simple(a, B)
    with pytest.raises(AssertionError):
        tmp.outer_real_row(a, B)
    with pytest.raises(AssertionError):
        tmp.outer_real_column(a, B)


def test_outer_real_compare_numpy_outer():
    'compare with numpy.outer'
    # set random generator
    rng = np.random.default_rng(555799917665544441234)
    vector_1 = rng.random(13)
    vector_2 = rng.random(13)
    reference_output_numpy = np.outer(vector_1, vector_2)
    computed_output_simple = tmp.outer_real_simple(vector_1, vector_2)
    computed_output_row = tmp.outer_real_row(vector_1, vector_2)
    computed_output_column = tmp.outer_real_column(vector_1, vector_2)
    aae(reference_output_numpy, computed_output_simple, decimal=10)
    aae(reference_output_numpy, computed_output_row, decimal=10)
    aae(reference_output_numpy, computed_output_column, decimal=10)


def test_outer_real_known_values():
    'check output produced by specific input'
    vector_1 = np.ones(5)
    vector_2 = np.arange(1,11)
    reference_output = np.resize(vector_2, (vector_1.size, vector_2.size))
    computed_output_simple = tmp.outer_real_simple(vector_1, vector_2)
    computed_output_row = tmp.outer_real_row(vector_1, vector_2)
    computed_output_column = tmp.outer_real_column(vector_1, vector_2)
    aae(reference_output, computed_output_simple, decimal=10)
    aae(reference_output, computed_output_row, decimal=10)
    aae(reference_output, computed_output_column, decimal=10)


def test_outer_real_transposition():
    'verify the transposition property'
    # set random generator
    rng = np.random.default_rng(555799917665544441234)
    a = rng.random(8)
    b = rng.random(5)
    a_outer_b_T_simple = tmp.outer_real_simple(a, b).T
    b_outer_a_simple = tmp.outer_real_simple(b, a)
    a_outer_b_T_row = tmp.outer_real_row(a, b).T
    b_outer_a_row = tmp.outer_real_row(b, a)
    a_outer_b_T_column = tmp.outer_real_column(a, b).T
    b_outer_a_column = tmp.outer_real_column(b, a)
    aae(a_outer_b_T_simple, b_outer_a_simple, decimal=10)
    aae(a_outer_b_T_row, b_outer_a_row, decimal=10)
    aae(a_outer_b_T_column, b_outer_a_column, decimal=10)


def test_outer_real_distributivity():
    'verify the distributivity property'
    rng = np.random.default_rng(111555799917665544441)
    a = rng.random(5)
    b = rng.random(5)
    c = rng.random(4)
    a_plus_b_outer_c_simple = tmp.outer_real_simple(a+b, c)
    a_outer_c_plus_b_outer_c_simple = (
        tmp.outer_real_simple(a, c) + tmp.outer_real_simple(b, c)
        )
    a_plus_b_outer_c_row = tmp.outer_real_row(a+b, c)
    a_outer_c_plus_b_outer_c_row = (
        tmp.outer_real_row(a, c) + tmp.outer_real_row(b, c)
        )
    a_plus_b_outer_c_column = tmp.outer_real_column(a+b, c)
    a_outer_c_plus_b_outer_c_column = (
        tmp.outer_real_column(a, c) + tmp.outer_real_column(b, c)
        )
    aae(a_plus_b_outer_c_simple, a_outer_c_plus_b_outer_c_simple, decimal=10)
    aae(a_plus_b_outer_c_row, a_outer_c_plus_b_outer_c_row, decimal=10)
    aae(a_plus_b_outer_c_column, a_outer_c_plus_b_outer_c_column, decimal=10)


def test_outer_real_scalar_multiplication():
    'verify scalar multiplication property'
    rng = np.random.default_rng(231115557999176655444)
    a = rng.random(3)
    b = rng.random(6)
    c = 3.4
    ca_outer_b = []
    a_outer_cb = []
    outer_real = {
        'simple' : outer_real_simple,
        'row' : outer_real_row,
        'column' : outer_real_column
    }
    for function in ['simple', 'row', 'column']:
        ca_outer_b.append(tmp.outer_real[function](c*a, b))
        a_outer_cb.append(tmp.outer_real[function](a, c*b))
    aae(ca_outer_b[0], a_outer_cb[0], decimal=10)
    aae(ca_outer_b[1], a_outer_cb[1], decimal=10)
    aae(ca_outer_b[2], a_outer_cb[2], decimal=10)


def test_outer_real_ignore_complex():
    'complex part of input must be ignored'
    vector_1 = np.ones(5) - 0.4j*np.ones(5)
    vector_2 = np.arange(1,11)
    reference_output = np.resize(vector_2, (vector_1.size, vector_2.size))
    outer_real = {
        'simple' : outer_real_simple,
        'row' : outer_real_row,
        'column' : outer_real_column
    }
    computed_output = []
    for function in ['simple', 'row', 'column']:
        computed_output.append(tmp.outer_real[function](vector_1, vector_2))
    aae(reference_output, computed_output[0], decimal=10)
    aae(reference_output, computed_output[1], decimal=10)
    aae(reference_output, computed_output[2], decimal=10)


def test_outer_complex_compare_numpy_outer():
    'compare hadamard_complex function with * operator'
    # for matrices
    rng = np.random.default_rng(876231115557999176655)
    input1 = rng.random(7) + 1j*rng.random(7)
    input2 = rng.random(7) + 1j*rng.random(7)
    output_numpy_outer = np.outer(input1, input2)
    output = []
    for function in ['simple', 'row', 'column']:
        output.append(tmp.outer_complex(vector_1, vector_2, function))
    aae(output[0], output_numpy_outer, decimal=10)
    aae(output[1], output_numpy_outer, decimal=10)
    aae(output[2], output_numpy_outer, decimal=10)


def test_outer_complex_invalid_function():
    'raise error for invalid function'
    for invalid_function in ['Simple', 'xxxxx', 'rows']:
        with pytest.raises(AssertionError):
            tmp.outer_complex(np.ones(3), np.ones(3), invalid_function)


# Hadamard product
def test_hadamard_real_different_shapes():
    'fail if input variables have different sizes'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        tmp.hadamard_real(a, B)


def test_hadamard_real_compare_asterisk():
    'compare hadamard_real function with * operator'
    # for vectors
    # set random generator
    rng = np.random.default_rng(11117665544444412)
    # use the random generator to create input parameters
    input1 = rng.random(18)
    input2 = rng.random(18)
    output = tmp.hadamard_real(input1, input2)
    output_asterisk = input1*input2
    aae(output, output_asterisk, decimal=10)
    # for matrices
    input1 = rng.random((5, 7))
    input2 = rng.random((5, 7))
    output = tmp.hadamard_real(input1, input2)
    output_asterisk = input1*input2
    aae(output, output_asterisk, decimal=10)


def test_hadamard_real_ignore_complex():
    'complex part of input must be ignored'
    # for vectors
    # set random generator
    rng = np.random.default_rng(9999999917665544444412)
    # use the random generator to create input parameters
    input1 = rng.random(10)
    input2 = rng.random(10) + 1j*np.ones(10)
    output = tmp.hadamard_real(input1, input2)
    output_reference = input1.real*input2.real
    aae(output, output_reference, decimal=10)
    # for matrices
    input1 = rng.random((5, 7)) - 1j*np.ones((5,7))
    input2 = rng.random((5, 7))
    output = tmp.hadamard_real(input1, input2)
    output_reference = input1.real*input2.real
    aae(output, output_reference, decimal=10)


def test_hadamard_complex_compare_asterisk():
    'compare hadamard_complex function with * operator'
    # for matrices
    # set random generator
    rng = np.random.default_rng(777799917665544444412)
    input1 = rng.random((4, 3))
    input2 = rng.random((4, 3))
    output = tmp.hadamard_complex(input1, input2)
    output_asterisk = input1*input2
    aae(output, output_asterisk, decimal=10)












