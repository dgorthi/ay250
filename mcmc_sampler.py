## My own Metropolis-Hastings algorithm, which is tested by sampling
## from a one dimensional Gaussian distribution.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def lnP(mu,sigma,x):
    """The posterior distribution from which samples will be drawn.
    
    """
    return (-0.5*(mu-x)**2/sigma**2)

def next(x):
    """The proposal density is a normal distribution of width 5
    (determined by hit and trial for this particular problem),
    centered around the sample with higher probability between the
    current and the previous.

    """
    return (np.random.normal(loc=x,scale=3))

def markov_chain():
    """Generates a markov chain to sample a given log likelihood function.

    """
    x0  = np.random.rand(1); x1  = next(x0)      #Starting point
    x,a = np.array([]),np.array([])
    
    for i in range(10**4):
        x1 = next(x0)
        a = np.append(a,min(1,np.exp(lnP(5,1,x1) - lnP(5,1,x0))))
        if a[i]==1:
            x = np.append(x,x1)
            x0 = x1 
            continue
        elif a[i]>np.random.ranf():
            x = np.append(x,x1)
        else:
            x = np.append(x,x0)
    print('Scale:%d\tMean acceptance ratio:%f\n'%(4,np.mean(a)))
    return x

x = markov_chain()
y = lnP(5,1,x)

f1,(ax1,ax2) = plt.subplots(2,1,sharex=True)
hist=ax1.hist(x,bins=100)
bins=(hist[1][1:]+hist[1][:-1])/2
ax1.plot(bins,mlab.normpdf(bins,5,1))
ax1.set_ylabel('Frequency')

ax2.plot(x,y,'.')
ax2.set_xlabel('x')
ax2.set_ylabel('LnP(x)')

f2,ax3 = plt.subplots(1,1)
ax3.plot(y)
ax3.set_xlabel('Step Number')
ax3.set_ylabel('Ln(P)')

plt.show()

## next_perform was written to check the performance of
## next(). Specifically, to check if the width of the gaussian
## affected the nature of samples drawn. Conclusion: Too large a width
## or too small a width both cause the next function to diverge, 5 was
## found to be optimal by hit and trial.
# def next_perform():
#     k = np.array([])
#     k = np.append(k,0)
#     for i in range(10000):
#         k = np.append(k,next(k[i]))
#     return k
