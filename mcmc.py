## My own Metropolis-Hastings algorithm, which is tested by sampling
## from a one dimensional Gaussian distribution.
import numpy as np
import matplotlib.pyplot as plt

# The posterior distribution from which samples will be drawn.
def lnP(mu,sigma,x):
    return (-0.5*(mu-x)**2/sigma**2)

# The proposal density is a uniform distribution of width 3 centered
# around the current sample.
def next(x):
    return (3*np.random.ranf()+x-1.5)

def markov_chain():
    x0 = 0      #Starting point
    while True:
        x1 = next(x0)
        a = min(1,np.exp(lnP(5,1,x0))/np.exp(lnP(5,1,x1)))
        
        if a>1 or a>np.random.ranf():
            x = np.append(x,x1)
        else:
            x = np.append(x,x0)
    
        x0 = x1
