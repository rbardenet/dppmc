# Import necessary libraries, basically random number generators, linear algebra, and plotting tools
import numpy as np 
import numpy.linalg as npl
import numpy.random as npr
import scipy.stats as spst
import math
import itertools as itt

class GradedOrder():
    """
    create an iterator for the graded lexicographic of order up to N in dimension d
    """
    def __init__(self, N, d):
        self.N = N
        self.d = d
        self.cpt = 0
        M = math.ceil(N**(1./d))
        L = ["itt.filterfalse(lambda x: not "+str(m)+"-1 in x, itt.product(range("+str(m)+"), repeat="+str(d)+"))" for m in range(1,M+1)] # The use of strings is to avoid lazy interpretation of m being an issue
        self.it = itt.chain(*[eval(l) for l in L])
        # We concatenate a list of iterators, one for each onion layer, lexicographically ordered

    def __iter__(self):
        return self

    def __next__(self):
        self.cpt += 1
        if self.cpt>self.N:
            raise StopIteration
        return self.it.__next__()

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
            print("Error: bound in rejection sampling is incorrect")
        if u < fStar/M:
            accepted = 1
        cpt += 1
    if not accepted:
        xStar = np.zeros(d)+1e-5*npr.randn()
    return xStar, accepted
