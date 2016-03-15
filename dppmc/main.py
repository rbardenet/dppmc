import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import dpp as dpplib
import tools

print("in main:", tools.jacobi(5,.2,.2,-.1))
npr.seed(1)
npr.seed(1)
numSamples = 100
mydpp = dpplib.DPP(numSamples,2,[[-.5,-.5], [0., .5]],"test2D")
numPoints = 100
xtest = [np.array([0.47518737, -0.98277171])]
computed = np.array([mydpp.w(x)*mydpp.CDKernel(x,x)*np.pi**2*np.prod(np.sqrt(1-x**2)) for x in xtest])
print("hey", computed,  np.exp(mydpp.logZ))
mydpp.sample()
#mydpp.save()

