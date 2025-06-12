import numpy as np
from scipy.linalg import dft

# Operacoes com vetores

### Produto escalar-vetor
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

def scalar_vec_complex(a, x, check_input=True):
    '''
    Compute the dot product of a is a complex number and x
    is a complex vector.

    Parameters
    ----------
    a : scalar
        Complex number.

    x : array 1D
        Complex vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Product of a and x.
    '''
    if check_input is True:
        assert isinstance(a, (complex, float, int)), 'a may be complex or scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    # Code here

    return result

### Dot product
def dot_real(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    # Check input here
    # Code here
    return result


def dot_complex(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    # Check input here
    # Complete here
    return result
