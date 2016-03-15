import pytest
import sys
import numpy.random as npr
sys.path.append("..")
import numpy as np
import scipy.integrate as spi
import scipy.special as sps
from dpp import DPP

def test_normsOfPolynomials():
    """
    make sure the products are computed as expected
    """
    mydpp = DPP(20, 2, [[-.5,-.5],[.4,.6]], "test")
    print("hey", spi.quad(lambda x: 1/np.sqrt(1-x**2) * sps.jacobi(2,-.5,-.5,monic=1)(x)**2, -1, 1)[0])
    expected = spi.quad(lambda x: 1/np.sqrt(1-x**2) * sps.jacobi(2,-.5,-.5,monic=1)(x)**2, -1, 1)[0] * spi.quad(lambda x: (1-x)**.4*(1+x)**.6*sps.jacobi(3,.4,.6,monic=1)(x)**2, -1, 1)[0]
    computed = mydpp.squaredNormsOfPolys[(2,3)]
    assert round(computed - expected, 5) == 0

def test_CDKernel():
    """
    compare to closed formula for the Chebyshev case
    """
    M = 6
    d = 2
    N = M**d
    p = [-.5, -.5]
    mydpp = DPP(N, 2, [p for _ in range(d)], "test")
    theta = np.pi*npr.rand()
    expected = (1/np.pi + 2/np.pi*(M/2. + .5*np.cos((M-1)*theta)*np.sin(M*theta)/np.sin(theta) - 1))**2
    x = np.cos(theta)*np.ones((d,))
    computed = mydpp.CDKernel(x, x)
    assert round(computed - expected, 5) == 0

def test_baseMeasure():
    """
    check base measure
    """
    N = 10
    mydpp = DPP(N, 2, [[-.5,-.5],[-.5,-.5]], "test")
    x = npr.rand(2)
    expected = 1./np.prod(np.sqrt(1-x**2))
    computed = mydpp.w(x)
    assert round(computed - expected, 5) == 0

def test_ChowsBound():
    """
    test bound on K*w/q in rejection sampling
    """
    npr.seed(1)
    mydpp = DPP(20, 2, [[-.5,-.5],[.4,.3]], "test")
    numPoints = 100
    xtest = [2*npr.rand(2) - 1 for _ in range(numPoints)]
    computed = np.array([mydpp.w(x)*mydpp.CDKernel(x,x)*np.pi**2*np.prod(np.sqrt(1-x**2)) for x in xtest])
    #print("hey", computed,  np.exp(mydpp.logZ))
    assert np.max( computed - np.exp(mydpp.logZ)) <= 0

def test_ChowsBound_2():
    """
    test bound on K*w/q in rejection sampling
    """
    npr.seed(1)
    numSamples = 100
    mydpp = DPP(numSamples,2,[[-.5,-.5], [0., .5]],"test2D")
    numPoints = 100
    xtest = [np.array([0.47518737, -0.98277171])]
    computed = np.array([mydpp.w(x)*mydpp.CDKernel(x,x)*np.pi**2*np.prod(np.sqrt(1-x**2)) for x in xtest])
    print("hey", computed,  np.exp(mydpp.logZ))
    assert np.max( computed - np.exp(mydpp.logZ)) <= 0



