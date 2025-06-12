import numpy as np
from scipy.linalg import toeplitz, circulant
import matplotlib.pyplot as plt

# Produto vetor-escalar
def scalar_vec_real(a,x,check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N.

    The code uses a simple "for" to iterate on the array.

    input
    -----------------
    a: scalar
        Real number

    x: 1D array
       Vector with N elements.

    returns
    ------------------
    y: 1D array
       Vector with N elements equal the product between a and x.

    '''
    if check_input is True:
        assert isinstance(a, (float, int)), 'a must be a scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    result = np.empty_like(x)
    for i in range(x.size):
        result[i] = a*x[i]

    return result

def scalar_vec_complex(a,x,check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is a complex number and x is a complex vector in C^N.

    input
    -----------------
    a: scalar
        Complex number

    x: 1D array
       Vector with N complex elements.

    returns
    ------------------
    y: 1D array
       Vector with N elements equal the product between a and x.

    '''
    if check_input is True:
        assert isinstance(a, (complex,float, int)), 'a must be a scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    result_real = a.real*x[:].real - a.imag*x[:].imag
    result_imag = a.real*x[:].imag + a.imag*x[:].real

    result = result_real + 1j*result_imag
    return result
