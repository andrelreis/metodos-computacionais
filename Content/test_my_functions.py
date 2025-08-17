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

# Dot test
def test_dot_real_not_1D_arrays():
    'fail due to input that is not 1D array'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        mf.dot_real(vector_1, vector_2)


def test_dot_real_different_sizes():
    'fail due to inputs having different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        mf.dot_real(vector_1, vector_2)


def test_dot_real_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output = mf.dot_real(vector_1, vector_2)
    aae(reference_output, computed_output, decimal=10)


def test_dot_real_compare_numpy_dot():
    'compare with numpy.dot'
    # set random generator
    rng = np.random.default_rng(12765)
    # use the random generator to create input parameters
    vector_1 = rng.random(13)
    vector_2 = rng.random(13)
    reference_output_numpy = np.dot(vector_1, vector_2)
    computed_output = mf.dot_real(vector_1, vector_2)
    aae(reference_output_numpy, computed_output, decimal=10)


def test_dot_real_commutativity():
    'verify commutativity'
    # set random generator
    rng = np.random.default_rng(555543127)
    # use the random generator to create input parameters
    a = rng.random(15)
    b = rng.random(15)
    # a dot b = b dot a
    output_ab = mf.dot_real(a, b)
    output_ba = mf.dot_real(b, a)
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
    output_a_bc = mf.dot_real(a, b + c)
    output_ab_ac = mf.dot_real(a, b) + mf.dot_real(a, c)
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
    output_c1a_c2b = mf.dot_real(c1*a, c2*b)
    output_c1c2_ab = c1*c2*mf.dot_real(a, b)
    aae(output_c1a_c2b, output_c1c2_ab, decimal=10)


def test_dot_complex_compare_numpy_dot():
    'compare dot_complex, numpy and numba with numpy.dot'
    # set random generator
    rng = np.random.default_rng(1111763412)
    # use the random generator to create input parameters
    vector_1 = rng.random(13) + 1j*rng.random(13)
    vector_2 = rng.random(13) + 1j*rng.random(13)
    output = mf.dot_complex(vector_1, vector_2)
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output, output_numpy_dot, decimal=10)


# Outer tests
def test_outer_real_input_not_vector():
    'fail with non-vector inputs'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        mf.outer_real_simple(a, B)
    with pytest.raises(AssertionError):
        mf.outer_real_row(a, B)
    with pytest.raises(AssertionError):
        mf.outer_real_column(a, B)


def test_outer_real_compare_numpy_outer():
    'compare with numpy.outer'
    # set random generator
    rng = np.random.default_rng(555799917665544441234)
    vector_1 = rng.random(13)
    vector_2 = rng.random(13)
    reference_output_numpy = np.outer(vector_1, vector_2)
    computed_output_simple = mf.outer_real_simple(vector_1, vector_2)
    computed_output_row = mf.outer_real_row(vector_1, vector_2)
    computed_output_column = mf.outer_real_column(vector_1, vector_2)
    aae(reference_output_numpy, computed_output_simple, decimal=10)
    aae(reference_output_numpy, computed_output_row, decimal=10)
    aae(reference_output_numpy, computed_output_column, decimal=10)


def test_outer_real_known_values():
    'check output produced by specific input'
    vector_1 = np.ones(5)
    vector_2 = np.arange(1,11)
    reference_output = np.resize(vector_2, (vector_1.size, vector_2.size))
    computed_output_simple = mf.outer_real_simple(vector_1, vector_2)
    computed_output_row = mf.outer_real_row(vector_1, vector_2)
    computed_output_column = mf.outer_real_column(vector_1, vector_2)
    aae(reference_output, computed_output_simple, decimal=10)
    aae(reference_output, computed_output_row, decimal=10)
    aae(reference_output, computed_output_column, decimal=10)


def test_outer_real_transposition():
    'verify the transposition property'
    # set random generator
    rng = np.random.default_rng(555799917665544441234)
    a = rng.random(8)
    b = rng.random(5)
    a_outer_b_T_simple = mf.outer_real_simple(a, b).T
    b_outer_a_simple = mf.outer_real_simple(b, a)
    a_outer_b_T_row = mf.outer_real_row(a, b).T
    b_outer_a_row = mf.outer_real_row(b, a)
    a_outer_b_T_column = mf.outer_real_column(a, b).T
    b_outer_a_column = mf.outer_real_column(b, a)
    aae(a_outer_b_T_simple, b_outer_a_simple, decimal=10)
    aae(a_outer_b_T_row, b_outer_a_row, decimal=10)
    aae(a_outer_b_T_column, b_outer_a_column, decimal=10)


def test_outer_real_distributivity():
    'verify the distributivity property'
    rng = np.random.default_rng(111555799917665544441)
    a = rng.random(5)
    b = rng.random(5)
    c = rng.random(4)
    a_plus_b_outer_c_simple = mf.outer_real_simple(a+b, c)
    a_outer_c_plus_b_outer_c_simple = (
        mf.outer_real_simple(a, c) + mf.outer_real_simple(b, c)
        )
    a_plus_b_outer_c_row = mf.outer_real_row(a+b, c)
    a_outer_c_plus_b_outer_c_row = (
        mf.outer_real_row(a, c) + mf.outer_real_row(b, c)
        )
    a_plus_b_outer_c_column = mf.outer_real_column(a+b, c)
    a_outer_c_plus_b_outer_c_column = (
        mf.outer_real_column(a, c) + mf.outer_real_column(b, c)
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
        'simple' : mf.outer_real_simple,
        'row' : mf.outer_real_row,
        'column' : mf.outer_real_column
    }
    for function in ['simple', 'row', 'column']:
        ca_outer_b.append(outer_real[function](c*a, b))
        a_outer_cb.append(outer_real[function](a, c*b))
    aae(ca_outer_b[0], a_outer_cb[0], decimal=10)
    aae(ca_outer_b[1], a_outer_cb[1], decimal=10)
    aae(ca_outer_b[2], a_outer_cb[2], decimal=10)


def test_outer_real_ignore_complex():
    'complex part of input must be ignored'
    vector_1 = np.ones(5) - 0.4j*np.ones(5)
    vector_2 = np.arange(1,11)
    reference_output = np.resize(vector_2, (vector_1.size, vector_2.size))
    outer_real = {
        'simple' : mf.outer_real_simple,
        'row' : mf.outer_real_row,
        'column' : mf.outer_real_column
    }
    computed_output = []
    for function in ['simple', 'row', 'column']:
        computed_output.append(outer_real[function](vector_1, vector_2))
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
        output.append(mf.outer_complex(input1, input2, function))
    aae(output[0], output_numpy_outer, decimal=10)
    aae(output[1], output_numpy_outer, decimal=10)
    aae(output[2], output_numpy_outer, decimal=10)


def test_outer_complex_invalid_function():
    'raise error for invalid function'
    for invalid_function in ['Vasco', 'xxxxx', 'rows']:
        with pytest.raises(ValueError):
            mf.outer_complex(np.ones(3), np.ones(3), function=invalid_function)


# Hadamard product
def test_hadamard_real_different_shapes():
    'fail if input variables have different sizes'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        mf.hadamard_real(a, B)


def test_hadamard_real_compare_asterisk():
    'compare hadamard_real function with * operator'
    # for vectors
    # set random generator
    rng = np.random.default_rng(11117665544444412)
    # use the random generator to create input parameters
    input1 = rng.random(18)
    input2 = rng.random(18)
    output = mf.hadamard_real(input1, input2)
    output_asterisk = input1*input2
    aae(output, output_asterisk, decimal=10)
    # for matrices
    input1 = rng.random((5, 7))
    input2 = rng.random((5, 7))
    output = mf.hadamard_real(input1, input2)
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
    output = mf.hadamard_real(input1, input2)
    output_reference = input1.real*input2.real
    aae(output, output_reference, decimal=10)
    # for matrices
    input1 = rng.random((5, 7)) - 1j*np.ones((5,7))
    input2 = rng.random((5, 7))
    output = mf.hadamard_real(input1, input2)
    output_reference = input1.real*input2.real
    aae(output, output_reference, decimal=10)


def test_hadamard_complex_compare_asterisk():
    'compare hadamard_complex function with * operator'
    # for matrices
    # set random generator
    rng = np.random.default_rng(777799917665544444412)
    input1 = rng.random((4, 3))
    input2 = rng.random((4, 3))
    output = mf.hadamard_complex(input1, input2)
    output_asterisk = input1*input2
    aae(output, output_asterisk, decimal=10)

### matrix-vector product

def test_matvec_real_input_doesnt_match():
    'fail when matrix columns doesnt match vector size'
    A = np.ones((5,4))
    x = np.ones(3)
    with pytest.raises(AssertionError):
        mf.matvec_real_simple(A, x)
    with pytest.raises(AssertionError):
        mf.matvec_real_dot(A, x)
    with pytest.raises(AssertionError):
        mf.matvec_real_columns(A, x)


def test_matvec_real_functions_compare_numpy_dot():
    'compare matvec_real_XXXX with numpy.dot'
    matrix = np.ones((5,4))
    vector = np.ones(4)
    output_simple = mf.matvec_real_simple(matrix, vector)
    output_dot = mf.matvec_real_dot(matrix, vector)
    output_columns = mf.matvec_real_columns(matrix, vector)
    output_numpy_dot = np.dot(matrix, vector)
    aae(output_simple, output_numpy_dot, decimal=10)
    aae(output_dot, output_numpy_dot, decimal=10)
    aae(output_columns, output_numpy_dot, decimal=10)


def test_matvec_real_functions_ignore_complex():
    'complex part of input must be ignored'
    matrix = 5*np.ones((5,4)) - 0.3j*np.ones((5,4))
    vector = 6*np.ones(4) + 2j*np.ones(4)
    output_simple = mf.matvec_real_simple(matrix, vector)
    output_dot = mf.matvec_real_dot(matrix, vector)
    output_columns = mf.matvec_real_columns(matrix, vector)
    output_reference = np.dot(matrix.real, vector.real)
    aae(output_simple, output_reference, decimal=10)
    aae(output_dot, output_reference, decimal=10)
    aae(output_columns, output_reference, decimal=10)


def test_matvec_complex_compare_numpy_dot():
    'compare matvec_complex with numpy.dot'
    matrix = 5*np.ones((5,4)) - 0.3j*np.ones((5,4))
    vector = 6*np.ones(4) + 2j*np.ones(4)
    output_simple = mf.matvec_complex(matrix, vector, function='simple')
    output_dot = mf.matvec_complex(matrix, vector, function='dot')
    output_columns = mf.matvec_complex(matrix, vector, function='columns')
    output_numpy_dot = np.dot(matrix, vector)
    aae(output_simple, output_numpy_dot, decimal=10)
    aae(output_dot, output_numpy_dot, decimal=10)
    aae(output_columns, output_numpy_dot, decimal=10)


def test_matvec_complex_invalid_function():
    'must raise error for invalid function'
    A = np.ones((5,4))
    x = np.ones(4)
    with pytest.raises(ValueError):
        mf.matvec_complex(A, x, check_input=True, function='invalid-function')
    with pytest.raises(ValueError):
        mf.matvec_complex(A, x, check_input=True, function='column')
    with pytest.raises(ValueError):
        mf.matvec_complex(A, x, check_input=True, function='Dot')


### matrix-matrix product

def test_matmat_real_input_doesnt_match():
    'fail when matrices dont match to compute the product'
    A = np.ones((3,3))
    B = np.ones((4,5))
    with pytest.raises(AssertionError):
        mf.matmat_real_simple(A, B, check_input=True)
    with pytest.raises(AssertionError):
        mf.matmat_real_dot(A, B, check_input=True)
    with pytest.raises(AssertionError):
        mf.matmat_real_rows(A, B, check_input=True)
    with pytest.raises(AssertionError):
        mf.matmat_real_columns(A, B, check_input=True)
    with pytest.raises(AssertionError):
        mf.matmat_real_outer(A, B, check_input=True)


def test_matmat_real_functions_compare_numpy_dot():
    'compare matmat_real_XXXX with numpy.dot'
    rng = np.random.default_rng(1234599999777772311155)
    matrix_1 = np.ones((3,3))
    matrix_2 = np.ones((3,5))
    output_simple = mf.matmat_real_simple(matrix_1, matrix_2)
    output_dot = mf.matmat_real_dot(matrix_1, matrix_2)
    output_rows = mf.matmat_real_rows(matrix_1, matrix_2)
    output_columns = mf.matmat_real_columns(matrix_1, matrix_2)
    output_outer = mf.matmat_real_outer(matrix_1, matrix_2)
    reference = np.dot(matrix_1, matrix_2)
    aae(output_simple, reference, decimal=10)
    aae(output_dot, reference, decimal=10)
    aae(output_rows, reference, decimal=10)
    aae(output_columns, reference, decimal=10)
    aae(output_outer, reference, decimal=10)


def test_matmat_real_functions_ignore_complex():
    'complex part of input must be ignored'
    rng = np.random.default_rng(7623012345999997777723)
    matrix_1 = np.ones((5,3))
    matrix_2 = 6*np.ones((3,5)) - 0.7j*np.ones((3,5))
    output_simple = mf.matmat_real_simple(matrix_1, matrix_2)
    output_dot = mf.matmat_real_dot(matrix_1, matrix_2)
    output_rows = mf.matmat_real_rows(matrix_1, matrix_2)
    output_columns = mf.matmat_real_columns(matrix_1, matrix_2)
    output_outer = mf.matmat_real_outer(matrix_1, matrix_2)
    reference = np.dot(matrix_1.real, matrix_2.real)
    aae(output_simple, reference, decimal=10)
    aae(output_dot, reference, decimal=10)
    aae(output_rows, reference, decimal=10)
    aae(output_columns, reference, decimal=10)
    aae(output_outer, reference, decimal=10)


def test_matmat_complex_compare_numpy_dot():
    'compare matmat_complex with numpy.dot'
    rng = np.random.default_rng(87900054312345999997777723)
    matrix_1 = 10*np.ones((5,3)) - 5j*np.ones((5,3))
    matrix_2 = 6*np.ones((3,5)) - 0.7j*np.ones((3,5))
    output_simple = mf.matmat_complex(matrix_1, matrix_2, function='simple')
    output_dot = mf.matmat_complex(matrix_1, matrix_2, function='dot')
    output_rows = mf.matmat_complex(matrix_1, matrix_2, function='rows')
    output_columns = mf.matmat_complex(matrix_1, matrix_2, function='columns')
    output_outer = mf.matmat_complex(matrix_1, matrix_2, function='outer')
    reference = np.dot(matrix_1, matrix_2)
    aae(output_simple, reference, decimal=10)
    aae(output_dot, reference, decimal=10)
    aae(output_rows, reference, decimal=10)
    aae(output_columns, reference, decimal=10)
    aae(output_outer, reference, decimal=10)


def test_matmat_complex_invalid_function():
    'must raise error for invalid function'
    A = np.ones((5,4))
    B = np.ones((4,3))
    with pytest.raises(ValueError):
        mf.matmat_complex(A, B, check_input=True, function='invalid-function')
    with pytest.raises(ValueError):
        mf.matmat_complex(A, B, check_input=True, function='column')
    with pytest.raises(ValueError):
        mf.matmat_complex(A, B, check_input=True, function='Dot')


## Test triangular matrices 
def test_triangular_input_doesnt_match():
    'fail when matrix columns doesnt match vector size'
    A = np.ones((5,5))
    x = np.ones(3)
    with pytest.raises(AssertionError):
        mf.matvec_triu_prod3(A, x)
    with pytest.raises(AssertionError):
        mf.matvec_triu_prod5(A, x)
    with pytest.raises(AssertionError):
        mf.matvec_tril_prod8(A, x)
    with pytest.raises(AssertionError):
        mf.matvec_tril_prod10(A, x)


def test_triangular_functions_compare_numpy_dot():
    'compare triangular matrix-vector with numpy.dot'
    matrix_U = np.triu(np.ones((4,4)))
    matrix_L = np.tril(np.ones((4,4)))
    vector = np.ones(4)
    
    output_triu3 = mf.matvec_triu_prod3(matrix_U, vector)
    output_triu5 = mf.matvec_triu_prod5(matrix_U, vector)
    output_dot_U = np.dot(matrix_U,vector)
    
    output_tril8 = mf.matvec_tril_prod8(matrix_L, vector)
    output_tril10 = mf.matvec_tril_prod10(matrix_L, vector)
    output_dot_L = np.dot(matrix_L,vector)
    
    aae(output_triu3, output_dot_U, decimal=10)
    aae(output_triu5, output_dot_U, decimal=10)
    aae(output_tril8, output_dot_L, decimal=10)
    aae(output_tril10, output_dot_L, decimal=10)


def test_triangular_functions_ignore_complex():
    'complex part of input must be ignored'
    matrix_U = np.triu(5*np.ones((4,4)) - 0.3j*np.ones((4,4)))
    matrix_L = np.tril(5*np.ones((4,4)) - 0.3j*np.ones((4,4)))
    vector = 6*np.ones(4) + 2j*np.ones(4)
    
    output_triu3 = mf.matvec_triu_prod3(matrix_U, vector)
    output_triu5 = mf.matvec_triu_prod5(matrix_U, vector)
    output_reference_U = np.dot(matrix_U.real, vector.real)

    output_tril8 = mf.matvec_tril_prod8(matrix_L, vector)
    output_tril10 = mf.matvec_tril_prod10(matrix_L, vector)
    output_reference_L = np.dot(matrix_L.real, vector.real)
    
    aae(output_triu3, output_reference_U, decimal=10)
    aae(output_triu5, output_reference_U, decimal=10)
    aae(output_tril8, output_reference_L, decimal=10)
    aae(output_tril10, output_reference_L, decimal=10)
























