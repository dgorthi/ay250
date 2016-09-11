## Program to generate fake data with m=5, b=-2 and then to fit this
## fake data using a MCMC sampler implemented using the
## Metropolis-Hastings algorithm.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import emcee
import corner

sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

def fake_data(m,b,sigma,N):
    """Generates N fake data points for the straight line of slope m and
    intercept b, with gaussian uncertainities for each data point.

    """
    x = np.random.uniform(-10,10,N)
    err = np.random.normal(loc=0,scale=sigma,size=N)
    y = m*x + b + err
    
    return x,y,err

def lnP(theta,x,y,sigma):
    """Generates the log likelihood probability of the given parameters
    (m,b) fitting the data, from the inverse squared deviation of the
    fit. Assuming flat priors. Formula: P(m,b|D) = P(D|m,b)P(m|b)P(b)

    """
    y_pred = theta[1]*x+theta[0]
    d = norm.logpdf(y,loc=y_pred,scale=sigma)
    return np.sum(d)
    
def next(theta):
    """The proposal density is a normal distribution of width 5
    (determined by hit and trial for this particular problem),
    centered around theta.

    """
    return np.random.normal(loc=theta,scale=[0.08,0.08])

def markov_chain(x,y,sigma):
    """Generates a markov chain to sample a given log likelihood
    function. Calls functions lnP() and next() to compute the log
    likelihood of a given parameter set and the generate the next
    parameter set, respectively.

    """
    Burn_in = 1000
    theta0 = np.random.rand(2)              #Starting point
    theta1 = next(theta0)
    Pcurr  = lnP(theta0,x,y,sigma)
    p = Pcurr
    theta,a= theta0,np.array([])

    #Burn in the Markov chain
    for i in range(Burn_in):
        theta1 = next(theta0)
        Pnext  = lnP(theta1,x,y,sigma)       
        acc = min(1,np.exp(Pnext-Pcurr))
        if acc==1:
            theta0 = theta1; Pcurr = Pnext
            continue
    
    for i in range(10**4):
        theta1 = next(theta0)
        Pnext  = lnP(theta1,x,y,sigma)       
        a = np.append(a,min(1,np.exp(Pnext-Pcurr)))
        if a[i]==1:
            theta = np.vstack((theta,theta1))
            p = np.append(p,Pnext)
            theta0 = theta1; Pcurr = Pnext
            continue
        elif a[i]>np.random.ranf():
            theta = np.vstack((theta,theta1))
            p = np.append(p,Pnext)
        else:
            theta = np.vstack((theta,theta0))
            p = np.append(p,Pcurr)
    print('Mean acceptance ratio:%f\n'%np.mean(a))
    return theta[1:],p[1:],a

m,b,sigma = 5,-2,2
x,y,err = fake_data(m,b,sigma,N=100)
theta,p,a = markov_chain(x,y,sigma)

#Using emcee
ndim,nwalkers = 2,200
theta0 = [np.random.ranf(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=[x,y,sigma])
sampler.run_mcmc(theta0,100)

#Fake Data
f1,ax = plt.subplots(1,1)
ax.errorbar(x,y,yerr=err,fmt='o')
ax.set_xlabel('x')
ax.set_ylabel('y')

#Fit contour plot
f2 = corner.corner(theta,range=[(-3,3),(3,7)],labels=('intercept','slope'),show_titles=True,plot_contours=True,truths=[-2.,5.])

#Diagonistic plots
f3,ax3 = plt.subplots(1,1)
ax3.plot(np.exp(p[0:500]),'.')
ax3.set_xlabel('Step Number')
ax3.set_ylabel('P(x)')

f4,(ax4,ax5) = plt.subplots(2,1)
ax4.plot(theta[:,0],np.exp(p),'.')
ax4.set_xlabel('intercept')
ax5.plot(theta[:,1],np.exp(p),'.')
ax5.set_xlabel('slope')

f5 = corner.corner(sampler.flatchain,labels=('intercept','slope'),show_titles=True,truths=[-2.,5.])

plt.show()



