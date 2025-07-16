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
        assert isinstance(a, (complex,float,int)), 'a must be a scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    result_real = a.real*x[:].real - a.imag*x[:].imag
    result_imag = a.real*x[:].imag + a.imag*x[:].real

    result = result_real + 1j*result_imag
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
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.size == y.size, 'x and y must have the same size'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert y.ndim == 1, 'y must have ndim = 1'
    
    result = 0
    for i in range(x.size): 
        result += x.real[i]*y.real[i]
    
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
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.size == y.size, 'x and y must have the same size'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert y.ndim == 1, 'y must have ndim = 1'
    
    result_real = dot_real(x.real, y.real) - dot_real(x.imag, y.imag)
    result_imag = dot_real(x.real, y.imag) + dot_real(x.imag, y.real)
    result = result_real + 1j*result_imag
    
    return result

# Outer product
def outer_real_simple(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert y.ndim == 1, 'y must have ndim = 1'


    N = x.size
    M = y.size

    result = np.empty((N,M),dtype=float)
    for i in range(N): 
        for j in range(M): 
            result[i,j] = x.real[i]*y.real[j]
            
    return result


def outer_real_row(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code use a single for to compute the rows of 
    the resultant matrix as a scalar-vector product.

    This code uses the function 'scalar_vec_real'.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert y.ndim == 1, 'y must have ndim = 1'

    N = x.size
    M = y.size

    result = np.zeros((N,M),dtype=float)
    for i in range(N):  
        result[i,:] = x.real[i]*y.real[:]

    return result


def outer_real_column(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code use a single for to compute the columns of 
    the resultant matrix as a scalar-vector product.

    This code uses the function 'scalar_vec_real'.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert y.ndim == 1, 'y must have ndim = 1'
        
    N = x.size
    M = y.size

    result = np.zeros((N,M),dtype=float)
    for j in range(M):  
        result[:,j] = x.real[:]*y.real[j]

    return result


def outer_complex(x, y, check_input=True, function='simple'):
    '''
    Compute the outer product of x and y, where x and y are complex vectors.

    Parameters
    ----------
    x, y : 1D arrays
        Complex vectors.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the outer_real function to be used. The possible
        values are 'simple', 'row' and 'column'.

    Returns
    -------
    result : 2D array
        Outer product of x and y.
    '''
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert y.ndim == 1, 'y must have ndim = 1'
    
    if function not in ['simple', 'row','column']:
        raise ValueError("invalid function {}".format(function))

    outer_real = {
        'simple' : outer_real_simple,
        'row' : outer_real_row,
        'column' : outer_real_column
    }


    result_real = outer_real[function](x.real,y.real) - outer_real[function](x.imag,y.imag)
    result_imag = outer_real[function](x.real,y.imag) + outer_real[function](x.imag,y.real)
    result = result_real + 1j*result_imag

    return result

# Hadamard product
def hadamard_real(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses a simple doubly nested loop to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''

    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.shape == y.shape, 'x and y must have the same shape'

    result = x.real*y.real
    return result


def hadamard_complex(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors or matrices having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    if check_input is True:
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert type(y) == np.ndarray, 'y must be a numpy array'
        assert x.shape == y.shape, 'x and y must have the same shape'

    result_real = x.real*y.real - x.imag*y.imag
    result_imag = x.real*y.imag + x.imag*y.real

    result = result_real + 1j*result_imag

    return result

## Operations with matrix

# Matrix-vector product
def matvec_real_simple(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code uses a simple doubly nested "for" to iterate on the arrays.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert A.shape[1] == x.shape[0], 'Number of columns A must have equal to the number of x elements'

    N,M = A.shape
    result = np.zeros(N)
    for i in range(N):
        for j in range(M):
            result[i] += A.real[i,j]*x.real[j]
    return result


def matvec_real_dot(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a dot product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert A.shape[1] == x.shape[0], 'Number of columns A must have equal to the number of x elements'

    N,M = A.shape
    result = np.zeros(N)
    for i in range(N):
        result[i] = np.dot(A.real[i,:],x.real[:])

    return result


def matvec_real_columns(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a scalar-vector product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert A.shape[1] == x.shape[0], 'Number of columns A must have equal to the number of x elements'

    N,M = A.shape
    result = np.zeros(N)
    for j in range(M):
        result[:] += A.real[:,j]*x.real[j]

    return result


def matvec_complex(A, x, check_input=True, function='dot'):
    '''
    Compute the matrix-vector product of an NxM matrix A and
    a Mx1 vector x.

    Parameters
    ----------
    A : array 2D
        NxM matrix.

    x : array 1D
        Mx1 vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the matvec_real function to be used. The possible
        values are 'simple', 'dot' and 'columns'.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert A.shape[1] == x.shape[0], 'Number of columns A must have equal to the number of x elements'

    if function not in ['simple', 'dot','columns']:
        raise ValueError("invalid function {}".format(function))

    matvec_real = {
        'simple' : matvec_real_simple,
        'dot' : matvec_real_dot,
        'columns' : matvec_real_columns
    }

    result_R = matvec_real[function](A.real,x.real) - matvec_real[function](A.imag,x.imag)
    result_I = matvec_real[function](A.real,x.imag) + matvec_real[function](A.imag,x.real)

    result = result_R + 1j*result_I
    return result

# matrix-matrix product
def matmat_real_simple(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code uses a simple triply nested "for" to iterate on the arrays.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(B) == np.ndarray, 'B must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert B.ndim == 2, 'x must have ndim = 1'
        assert A.shape[1] == B.shape[0], 'Number of columns A must have equal to the number of rows of B'

    N = A.shape[0]
    M = B.shape[1]
    L = A.shape[1]
    result = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            for k in range(L):
                result[i,j] += A.real[i,k]*B.real[k,j]
    return result


def matmat_real_dot(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces one "for" by a dot product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(B) == np.ndarray, 'B must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert B.ndim == 2, 'x must have ndim = 1'
        assert A.shape[1] == B.shape[0], 'Number of columns A must have equal to the number of rows of B'

    N = A.shape[0]
    M = B.shape[1]
    result = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            result[i,j] = np.dot(A.real[i,:],B.real[:,j])

    return result


def matmat_real_rows(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by a matrix-vector product defining
    a row of the resultant matrix.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(B) == np.ndarray, 'B must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert B.ndim == 2, 'x must have ndim = 1'
        assert A.shape[1] == B.shape[0], 'Number of columns A must have equal to the number of rows of B'

    N = A.shape[0]
    M = B.shape[1]
    result = np.zeros((N,M))
    for i in range(N):
        result[i,:] = B.real[:,:].T@A.real[i,:]

    return result


def matmat_real_columns(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by a matrix-vector product defining
    a column of the resultant matrix.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(B) == np.ndarray, 'B must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert B.ndim == 2, 'x must have ndim = 1'
        assert A.shape[1] == B.shape[0], 'Number of columns A must have equal to the number of rows of B'

    N = A.shape[0]
    M = B.shape[1]
    result = np.zeros((N,M))
    for j in range(M):
        result[:,j] = A.real[:,:]@B.real[:,j]
    return result


def matmat_real_outer(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by an outer product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(B) == np.ndarray, 'B must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert B.ndim == 2, 'x must have ndim = 1'
        assert A.shape[1] == B.shape[0], 'Number of columns A must have equal to the number of rows of B'

    N = A.shape[0]
    M = B.shape[1]
    L = A.shape[1]
    result = np.zeros((N,M))
    for k in range(L):
        result[:,:] += np.outer(A.real[:,k],B.real[k,:])
    return result


def matmat_complex(A, B, check_input=True, function='simple'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in C^NxM and B in C^MxP.

    Parameters
    ----------
    A, B : 2D arrays
        Complex matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the matmat_real function to be used. The possible
        values are 'simple', 'dot', 'rows', 'columns' or 'outer'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    if check_input is True:
        assert type(A) == np.ndarray, 'A must be a numpy array'
        assert type(B) == np.ndarray, 'B must be a numpy array'
        assert A.ndim == 2, 'A must have ndim = 2'
        assert B.ndim == 2, 'x must have ndim = 1'
        assert A.shape[1] == B.shape[0], 'Number of columns A must have equal to the number of rows of B'

    if function not in ['simple', 'dot','rows','columns','outer']:
        raise ValueError("invalid function {}".format(function))

    

    matmat_real = {
        'simple' : matmat_real_simple,
        'dot' : matmat_real_dot,
        'rows' : matmat_real_rows,
        'columns' : matmat_real_columns,
        'outer' : matmat_real_outer
    }


    result_R = matmat_real[function](A.real,B.real) - matmat_real[function](A.imag,B.imag)
    result_I = matmat_real[function](A.real,B.imag) + matmat_real[function](A.imag,B.real)

    result = result_R + 1j*result_I
    return result

## Triangular matrices
def matvec_triu_prod3(U, x, check_input=True):
    '''
    Compute the product of an upper triangular matrix U 
    and a vector x. All elements are real numbers.
    
    Each element of the resultant vector is obtained by 
    computing a dot product.

    Parameters
    ----------
    U : numpy array 2d
        Upper triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''
    if check_input is True:
        assert type(U) == np.ndarray, 'U must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert U.ndim == 2, 'U must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert U.shape[1] == x.shape[0], 'Number of columns U must have equal to the number of x elements'
        assert U.shape[0] == U.shape[1], 'U must be a square matrix'
    
    N = U.shape[0]
    result = np.zeros(N)
    for i in range(N):
        result[i] = np.dot(U.real[i,i:],x.real[i:])
    
    return result

def matvec_triu_prod5(U, x, check_input=True):
    '''
    Compute the product of an upper triangular matrix U 
    and a vector x. All elements are real numbers.
    
    The elements of the resultant vector are obtained by 
    computing successive scalar vector products.

    Parameters
    ----------
    U : numpy array 2d
        Upper triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''
    if check_input is True:
        assert type(U) == np.ndarray, 'U must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert U.ndim == 2, 'U must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert U.shape[1] == x.shape[0], 'Number of columns U must have equal to the number of x elements'
        assert U.shape[0] == U.shape[1], 'U must be a square matrix'

    N = U.shape[0]
    result = np.zeros(N)
    for j in range(N):
        result[:j+1] += x.real[j]*U.real[:j+1,j]
    
    return result

def matvec_tril_prod8(L, x, check_input=True):
    '''
    Compute the product of an lower triangular matrix L 
    and a vector x. All elements are real numbers.
    
    Each element of the resultant vector is obtained by 
    computing a dot product.

    Parameters
    ----------
    L : numpy array 2d
        Lower triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''
    if check_input is True:
        assert type(L) == np.ndarray, 'L must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert L.ndim == 2, 'L must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert L.shape[1] == x.shape[0], 'Number of columns L must have equal to the number of x elements'
        assert L.shape[0] == L.shape[1], 'L must be a square matrix'

    N = L.shape[0]
    result = np.zeros(N)
    for i in range(N):
        print (L[i,:i+1])
        result[i] = np.dot(L.real[i,:i+1],x.real[:i+1])
    
    return result

def matvec_tril_prod10(L, x, check_input=True):
    '''
    Compute the product of an lower triangular matrix L 
    and a vector x. All elements are real numbers.
    
    The elements of the resultant vector are obtained by 
    computing successive scalar vector products.

    Parameters
    ----------
    L : numpy array 2d
        Lower triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''
    if check_input is True:
        assert type(L) == np.ndarray, 'L must be a numpy array'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert L.ndim == 2, 'L must have ndim = 2'
        assert x.ndim == 1, 'x must have ndim = 1'
        assert L.shape[1] == x.shape[0], 'Number of columns L must have equal to the number of x elements'
        assert L.shape[0] == L.shape[1], 'L must be a square matrix'

    N = L.shape[0]
    result = np.zeros(N)
    for j in range(N):
        result[j:] += x.real[j]*L.real[j:,j]
    
    return result




























