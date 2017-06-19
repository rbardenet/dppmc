import numpy as np 
import numpy.linalg as npl
import numpy.random as npr
import scipy.stats as spst
import itertools as itt
import pickle as pkl
import scipy.integrate as spi
import scipy.special as sps
import progressbar as pb
from scipy.misc import logsumexp
from tools import jacobi, GradedOrder, schurInversion, rejectionSamplingWithBetaProposal

class DPP:
    """
    implement a determinantal point process, with a method to sample from it
    """

    def __init__(self, N, dimension, listOfParams, jobId):
        """
        dimension: dimension of the ambient space
        params: d-list of 2-lists of parameters [[a1,b1], [a2,b2], ...] of the involved Jacobi polynomials
        """
        self.N = N
        self.d = dimension
        self.a, self.b = zip(*listOfParams)
        self.jobId = jobId
        self.numTrials = 10000 # number of trials in rejection sampling
        self.squaredNormsOfPolys = {} # keys will be multiindices
        self.computeSquaredNormsOfPolys()
        self.logZ = 0

        self.checkParams()
        self.computeChowsBound() # uniform bound on base measure times diagonal kernel, useful for rejection sampling later

    def checkParams(self):
        """
        check that the parameters are valid
        """
        a = np.array(self.a)
        b = np.array(self.b)
        if (a<-.5).any() or (a>.5).any() or (b<-.5).any() or (b>.5).any():
            print("Error: our bounds in rejection sampling are invalid")
        return 1

    def computeSquaredNormsOfPolys(self):
        """
        compute L2 norms of all involved Jacobi polynomials, as we need orthonormal polynomials
        """
        a, b = self.a, self.b
        for k in GradedOrder(self.N, self.d):
            self.squaredNormsOfPolys[k] = 1.0*np.prod([spi.quad(lambda x: (1-x)**a[i]*(1+x)**b[i]*jacobi(k[i],a[i],b[i],x)**2, -1, 1)[0] for i in range(self.d)])

    def computeChowsBound(self):
        """
        bound on w*K/q taken from [Chow, Gatteschi, and Wong, 1994]
        this is useful for rejection sampling
        TODO: implement in logsumexp style
        """
        a, b = self.a, self.b
        res = []
        for k in GradedOrder(self.N, self.d):
            logacc = 0
            for i in range(self.d):
                #logacc = 0 # attention!
                kk = k[i]
                if kk == 0:
                    cst = spi.quad(lambda x: (1-x)**a[i]*(1+x)**b[i], -1, 1)[0]
                    logacc -= np.log(cst) # Phi_0^2 = 1/cst
                    if not (a[i]==-.5 and b[i]==-.5):
                        # We're not in the Chebyshev case
                        mode = (b[i]-a[i])*1./(a[i]+b[i]+1)
                        logacc += (a[i]+.5)*np.log(1-mode) + (b[i]+.5)*np.log(1+mode) + np.log(np.pi)
                    else:
                        # In the Chebyshev case, w/q = pi
                        logacc += np.log(np.pi)
                        #print("logacc", logacc)
                else:
                    # Use Chow et al's bound
                    aa = min(a[i],b[i])
                    bb = max(a[i],b[i])
                    logacc += np.log(2) + sps.gammaln(kk+aa+bb+1) + sps.gammaln(kk+bb+1) - sps.gammaln(kk+aa+1) - sps.gammaln(kk+1) - 2*bb*np.log(kk+(aa+bb+1.)/2)
                #print("ho", i, k, np.exp(logacc))

            res.append(logacc)
        self.logZ = logsumexp(res)

    def CDKernel(self, x, y):
        """
        compute Christoffel-Darboux kernel
        """
        a, b = self.a, self.b
        res = 0
        for k in GradedOrder(self.N, self.d):
            #print("in kernel", k, [(1-x[i])**(a[i]+.5)*(1+x[i])**(b[i]+.5)*np.pi*sps.jacobi(k[i],a[i],b[i],monic=1)(x[i])**2/(spi.quad(lambda t:(1-t)**(a[i])*(1+t)**(b[i])*sps.jacobi(k[i],a[i],b[i],monic=1)(t)**2, -1,1)[0]) for i in range(self.d)])
            
            inc = np.prod([jacobi(k[i],a[i],b[i],x[i])*jacobi(k[i],a[i],b[i],y[i]) for i in range(self.d)])
            inc /= self.squaredNormsOfPolys[k]
            #print("inc", self.w(x)*np.pi**2*np.prod(np.sqrt(1-x**2))*inc)
            res += inc
        return res

    def w(self, x):
        """
        pdf of base measure
        """
        a, b = self.a, self.b
        return np.prod([(1-x[i])**a[i]*(1+x[i])**b[i] for i in range(self.d)])

    def sample(self):
        N = self.N
        d = self.d
        self.X = np.zeros((N, d))
        K = lambda x,y: self.CDKernel(x, y)
        f = lambda x: K(x,x)*self.w(x)/N # initialize to intensity measure

        numBarLevels = 20
        bar = pb.ProgressBar(max_value=numBarLevels)
        # Draw the first point from the intensity measure
        self.X[N-1,:], failed = rejectionSamplingWithBetaProposal(f, 1./N*np.exp(self.logZ), d, self.numTrials) # 1. is an upper bound on f
        if failed:
            print("failed RS")
        D = np.array([(K(self.X[N-1,:], self.X[N-1,:]))]).reshape((1,1))
        invK = npl.inv(D)

        # Draw all subsequent points from the right conditional
        for i in reversed(range(N-1)):
            bar.update(int((N-i)/N*numBarLevels))
            #if not np.mod(i, N/5):
            # from time to time print where we are in the loop

            # Define conditional
            xx = [self.X[j,:] for j in range(i+1,N)]
            #print("hey", xx)

            def f(x):
                kk = np.array(list(map(lambda y: K(y,x), xx))).reshape((1,N-i-1))
                return 1.0/(i+1)*(K(x,x) - np.dot(kk, np.dot(invK, kk.T)))*self.w(x)

            #f = lambda x:  1.0/(i+1)* ( K(x,x) - np.dot(np.array(list(map(lambda y: K(y,x), xx))).reshape((1,N-i-1)), np.dot(invK, \
            #                np.array(list(map(lambda y: K(x,y), xx))).reshape((N-i-1,1)) ) ))*self.w(x) # this comes from the normal equations

            # Draw next point using rejection sampling
            self.X[i,:], failed = rejectionSamplingWithBetaProposal(f, 1./(i+1)*np.exp(self.logZ), d, self.numTrials)
            if failed:
                print("Error: Rejection sampling failed")

            # Use Schur inversion for computational efficiency
            C = np.array([K(self.X[i,:], self.X[j,:]) for j in range(i+1,N)]).reshape((N-i-1,1))
            invK = schurInversion(np.array(K(self.X[i,:], self.X[i,:])).reshape((1,1)), C.T, C, invK)
        
    def save(self):
        pkl.dump(self, open(self.jobId+".pkl", "wb"))
