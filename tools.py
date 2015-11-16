# Import necessary libraries, basically random number generators, linear algebra, and plotting tools
import numpy as np 
import numpy.linalg as npl
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as spst
import itertools as itt
# Change the size of labels on figures
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)

def realify(z):
    """
    set z to its real part and raise a flag ig imag part is large
    """
    if np.imag(z)>1e-5:
        print "Error: nonzero imaginary part", "Im(z)=", np.imag(z)
    return np.real(z)

def schurInversion(A, B, C, invD):
    """
    invert block matrix (A, B; C, D) when you know the inverse of D
    """
    size = len(invD) + len(A)
    res = np.zeros((size, size), dtype=invD.dtype)
    S = A - np.dot(np.dot(B, invD), C) # Schur complement
    invS = npl.inv(S)
    BinvD = np.dot(B, invD)
    invDC = np.dot(invD, C)
    res[:len(A), :len(A)] = invS
    res[:len(A), len(A):] = -np.dot(invS, BinvD)
    res[len(A):, :len(A)] = -np.dot(invDC, invS)
    res[len(A):, len(A):] = invD + np.dot(np.dot(invDC, invS), BinvD)
    return res

def rejectionSamplingWithUniformProposal(f, Z, d, maxTrials=1000):
    """
    implements generic rejection sampling on [0,1]^d
    f: target pdf
    Z: upper bound on f
    d: dimension
    maxTrials: max number of steps in RS
    """
    accepted = 0
    cpt = 0
    M = Z # this is an UB on f/q
    while not accepted and cpt<maxTrials:
        xStar = npr.rand(d)
        u = npr.rand()
        fStar = f(xStar)
        if fStar/M > 1:
            print "Error: bound in rejection sampling is incorrect"
        if u < fStar/M:
            accepted = 1
        cpt += 1
    if not accepted:
        xStar = np.zeros(d)+1e-5*npr.randn()
    return xStar, accepted
