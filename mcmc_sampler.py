## My own Metropolis-Hastings algorithm, which is tested by sampling
## from a one dimensional Gaussian distribution.
import numpy as np
import matplotlib.pyplot as plt


def lnP(mu,sigma,x):
    """The posterior distribution from which samples will be drawn."""
    return (-0.5*(mu-x)**2/sigma**2)

def next(x):
    """The proposal density is a normal distribution of width 1, centered
    around the sample with higher probability between the current and
    the previous."""
    return (np.random.normal(loc=x,scale=5))

def markov_chain():
    x0  = 0; x1  = next(x0)      #Starting point
    x,a = np.array([]),np.array([])
    
    for i in range(10**5):
        a = np.append(a,min(1,np.exp(lnP(5,1,x1) - lnP(5,1,x0))))
        if a[i]==1:
            x = np.append(x,x1)
            x0 = x1; x1 = next(x1)
            continue
        elif a[i]>np.random.ranf():
            x = np.append(x,x1)
        else:
            x = np.append(x,x0)
        tmp = x1
        x1 = next(x0)
        x0 = tmp
    print('Mean acceptance ratio:%f\n'%np.mean(a))
    return x

f,ax = plt.subplots(1,1)
ax.hist(markov_chain(),bins=np.linspace(0,10,21))
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
#ax.plot(x)

plt.show()

## next_perform was written to check the performance of
## next(). Specifically, to check if the width of the gaussian
## affected the nature of samples drawn. Conclusion: Too large a width
## or too small a width both cause the next function to diverge, 1 was
## found to be optimal by hit and trial.
# def next_perform():
#     k = np.array([])
#     k = np.append(k,0)
#     for i in range(10000):
#         k = np.append(k,next(k[i]))
#     return k
