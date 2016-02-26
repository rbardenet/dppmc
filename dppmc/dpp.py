import numpy as np 
import numpy.linalg as npl
import numpy.random as npr
import scipy.stats as spst
import itertools as itt
import matplotlib.pyplot as plt
from tools import schurInversion, rejectionSamplingWithUniformProposal 

class DPP: 
    """
    implement a determinantal point process, with a method to sample from it
    """

    def __init__(self, dimension, params):
        """
        dimension: dimension of the ambient space
        params: list of lists of parameters [[a1,b1], [a2,b2], ...] of the involved Jacobi polynomials
        """
        self.d = dimension
        self.params = params
        self.numTrials = 10000 # number of trials in rejection sampling
        self.multiInd = np.zeros((N,2), int)
        self.computeMultiIndices()
        print(">> Initialized DPP")

    def checkParams(self):
        """
        check that the parameters are valid
        """
        return 1

    def kernel(self, x, y):
        """
        evaluate the kernel of the DPP
        """
        d = self.d

    def kernel_per_app(self, x, y):
        """
        evaluate periodic approximate kernel, as defined in [LaMoRu12]
        """
        d = self.d
        Phi = lambda k, u: np.exp(2*1j*np.pi*np.dot(k,u))
        if self.kernelName == "Gaussian":
             res = np.sum([self.phi(k)*Phi(k, x-y) for k in self.inds])
             return realify(res)
        else:
            print "Error: undefined kernel"

    def compareKernels(self):
        """
        compare real and approximate kernels
        """
        nPlot = 30
        y = np.linspace(-1.0, 1.0, nPlot)
        rho, alpha = self.params
        myLevels = np.arange(-rho*0.1,rho*1.1,rho/100.) # Colors on both plots should be comparable
        cmap = "seismic" # play with matplotlib's colormaps...

        # plot original kernel
        plt.subplot(131)
        Z = np.array([[self.kernel(np.array([0,0]), np.array([y[i], y[j]])) for j in range(nPlot)] for i in range(nPlot)])
        plt.contourf(y, y, Z, levels=myLevels, cmap=cmap)
        plt.title("Original kernel")

        # plot approximate kernel
        plt.subplot(132)
        Z_approx = np.array([[self.kernel_per_app(np.array([0,0]), np.array([y[i], y[j]])) for j in range(nPlot)] for i in range(nPlot)])
        plt.contourf(y, y, Z_approx, levels=myLevels, cmap=cmap)
#        ax = plt.gca()
#        ax.add_patch(patches.Rectangle((-.5, -.5), 1, 1, fill=False, linewidth=5, color="white"))
        plt.title("approx. kernel")

        # plot absolute difference with the same scale
        plt.subplot(133)
        plt.contourf(y, y, np.abs(Z-Z_approx), levels=myLevels, cmap=cmap)
        plt.colorbar()
        plt.axvline(.5, color='w', linewidth=3)
        plt.axvline(-.5, color='w', linewidth=3)
        plt.axhline(.5, color='w', linewidth=3)
        plt.axhline(-.5, color='w', linewidth=3)

#        ax = plt.gca()
#        ax.vline()
#       ax.add_patch(patches.Rectangle((-.5, -.5), 1, 1, fill=False, linewidth=5, color="white"))
        plt.title("Absolute error")

        plt.show()

    def sampleBernoullis(self):
        """
        first step of DPP sampling, see [HoKrPeVi06]
        """
        d = self.d
        if self.kernelName == "Gaussian":
            eigs = [self.phi(k) for k in self.inds]
            self.berns = npr.binomial(1,eigs)
            self.N = np.sum(self.berns)
            print ">> Sampled Bernoullis, number of ones is N=", self.N 
        
    def KTilde(self, x, y):
        if self.kernelName == "Gaussian":
            Phi = lambda k, u: np.exp(2*1j*np.pi*np.dot(k,u))
        return np.sum([Phi(k, x-y) for k in self.inds[np.where(self.berns)]])

    def w(self, x):
        """
        base measure is the indicator of [0,1]^d
        """
        return 1.*np.all(x>0)*np.all(x<1)
    
    def sample(self):
        N = self.N
        d = self.d
        self.X = np.zeros((N, d))
        K = lambda x,y: self.KTilde(x, y)
        f = lambda x: realify(K(x,x))*self.w(x)/N # initialize to intensity measure
        diagK = np.zeros((N,))
        
        # Draw the first point from the intensity measure
        print "Sampling the", N, "th point"

        self.X[N-1,:], success = rejectionSamplingWithUniformProposal(f, 1., d, self.numTrials) # 1. is an upper bound on f
        D = np.array([(K(self.X[N-1,:], self.X[N-1,:]))]).reshape((1,1))
        invK = npl.inv(D)
        
        # Draw all subsequent points from the right conditional
        for i in reversed(range(N-1)):
 
            if not np.mod(i, N/5):
                # from time to time print where we are in the loop
                print "Sampling the", i, "th point"
            
            # Define conditional
            xx = [self.X[j,:] for j in range(i+1,N)]

            f = lambda x:  1.0/(i+1)* realify( K(x,x) - np.dot(np.array(map(lambda y: K(y,x), xx)).reshape((1,N-i-1)), np.dot(np.conjugate(invK), \
                                                                np.array(map(lambda y: K(x,y), xx)).reshape((N-i-1,1))) ) )*self.w(x) # this comes from the normal equations
                                                    
            # Draw next point using rejection sampling
            self.X[i,:], success = rejectionSamplingWithUniformProposal(f, N*1./(i+1), d, self.numTrials)
            if not success:
                print "Error: Rejection sampling failed"

            # Save first conditional for later plotting
            if i==N-2:
                print "Saving the first conditional"
                nPlot = 60
                self.yPlotFirstCond = np.linspace(0., 1.0, nPlot)
                self.ZPlotFirstCond = [f(np.array([self.yPlotFirstCond[ii], self.yPlotFirstCond[jj]])) for jj in range(nPlot) for ii in range(nPlot)]

            # Save last conditional for later plotting
            if i==0:
                print "Saving the last conditional"
                nPlot = 40
                self.yPlotLastCond = np.linspace(0., 1.0, nPlot)
                self.ZPlotLastCond = [f(np.array([self.yPlotLastCond[ii], self.yPlotLastCond[jj]])) for jj in range(nPlot) for ii in range(nPlot)]

            # Use Schur inversion for computational efficiency
            C = np.array([K(self.X[i,:], self.X[j,:]) for j in range(i+1,N)]).reshape((N-i-1,1))
            invK = schurInversion(np.array(K(self.X[i,:], self.X[i,:])).reshape((1,1)), C.T, np.conjugate(C), invK) # Beware that K is Hermitian, not symmetric
                
        print ">> Done"

    def plotConditionals(self):
        """
        plot first and last conditionals during sampling
        """
        cmap = "seismic"

        # plot first conditional
        plt.subplot(121)
        yPlot = self.yPlotFirstCond
        nPlot = len(yPlot)
        ZPlot = self.ZPlotFirstCond
        plt.contourf(yPlot, yPlot, np.array(ZPlot).reshape((nPlot,nPlot)), cmap=cmap, alpha=.8)
        plt.colorbar()
        plt.plot(self.X[-1,0], self.X[-1,1], 'o', markersize=8, color="lightgreen") # First point sampled
        plt.plot(self.X[-2,0], self.X[-2,1], '*', markersize=22, color="yellow", markeredgewidth=2) # Second point sampled
        plt.axvline(.25, color='black', linewidth=3, alpha=.5)
        plt.axvline(.75, color='black', linewidth=3, alpha=.5)
        plt.axhline(.25, color='black', linewidth=3, alpha=.5)
        plt.axhline(.75, color='black', linewidth=3, alpha=.5)
        
        # plot last conditional
        plt.subplot(122)
        yPlot = self.yPlotLastCond
        nPlot = len(yPlot)
        ZPlot = self.ZPlotLastCond
        plt.contourf(yPlot, yPlot, np.array(ZPlot).reshape((nPlot,nPlot)), cmap=cmap, alpha=.8)
        plt.colorbar()
        plt.plot(self.X[1:,0], self.X[1:,1], 'o', markersize=8, color="lightgreen") # Previous points sampled
        plt.plot(self.X[0,0], self.X[0,1], '*', markersize=22, color="yellow", markeredgewidth=2) # Last point sampled
        plt.axvline(.25, color='black', linewidth=3, alpha=.5)
        plt.axvline(.75, color='black', linewidth=3, alpha=.5)
        plt.axhline(.25, color='black', linewidth=3, alpha=.5)
        plt.axhline(.75, color='black', linewidth=3, alpha=.5)
        
        plt.show()
