import sys
sys.path.append("..")
import dpp as dpplib
import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spst

# Setting parameters
print(">>> Welcome. In this example we sample a 2D orthogonal polynomial ensemble")
p = [[-.2,+.2], [0., .5]]

# Setting a seed for reproducibility
npr.seed(3)

# Sampling the DPP
numSamples = 50
mydpp = dpplib.DPP(numSamples,2,p,"test2D")
mydpp.sample()
mydpp.save() # This will create a pkl object with your Sample

# Plotting the sample
X = np.array(mydpp.X)
g = sns.jointplot(X[:,0], X[:,1], marginal_kws=dict(bins=10))
xplot = np.linspace(-1.,1.,100)
aa = p[0][0]
bb = p[0][1]
g.ax_marg_x.plot(xplot, 2**(-aa-bb+3)*spst.beta(aa+1, bb+1).pdf((1+xplot)/2))
g.ax_marg_x.plot(xplot, 2**(3)*spst.beta(.5, .5).pdf((1+xplot)/2))
aa = p[1][0]
bb = p[1][1]
g.ax_marg_y.plot(2**(-aa-bb+3)*spst.beta(aa+1, bb+1).pdf((1+xplot)/2), xplot)
g.ax_marg_y.plot(2**(3)*spst.beta(.5, .5).pdf((1+xplot)/2), xplot)
plt.savefig("example1.png")
print(">>> I created example1.png, which you should be prompted right now as well")
plt.show()
