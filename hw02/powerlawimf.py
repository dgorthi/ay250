import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from   matplotlib import rc
import matplotlib
from   scipy import integrate
from   scipy.stats import norm
import emcee
import corner

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=3)

## Throughout this code:
## (1) alpha is negative
## (2) theta = [Mmax,alpha]

def random_masses(a,Mmin,Mmax,N=1):
    return (Mmin**(a+1)+(Mmax**(a+1)- Mmin**(a+1))*np.random.random(size=N))**(1/(a+1))

def imf(m,alpha):
    return m**(alpha)

# def lnprior(theta,Mmin):
#     Mmax,alpha = theta[0],theta[1]
#     if (-5 < alpha < 0) and (Mmax>Mmin):
#         return 0
#     return -np.inf

def lnP(theta,mass,num):
    """Generates the log likelihood probability of the given parameters
    (m,b) fitting the data, from the inverse squared deviation of the
    fit. Assuming flat priors. Formula: P(m,b|D) = P(D|m,b)P(m|b)P(b)

    """
    n_pred = theta[1]*mass +theta[0]
    d = norm.logpdf(num,loc=n_pred,scale=0.01)
    return np.sum(d)

# def lnP(theta,mass,num):
#     num_pred =  theta[1]*mass + theta[0]
#     return -np.sum((num_pred- num)**2)

# def lnP(theta,Mmin,mass,num):
#     lp = lnprior(theta,Mmin)
#     if not np.isfinite(lp):
#         return -np.inf
#     Mmax, alpha = theta[0],theta[1]
#     c,err = integrate.quad(imf,Mmin,Mmax,args=alpha)
#     if c<=0:
#         print ("lnP(): c<0 encountered")
#         return -np.inf
#     n_pred = alpha*mass-np.log(c)
#     logP = lp-np.sum((num-n_pred)**2)
#     if np.isfinite(logP):
#         return logP
#     print ("lnP(): non-finite logP encountered")
#     return -np.inf

def mcmc(ndim,nwalkers,mass,num):
    """Use emcee to fit a line to the Salpeter IMF data in log space."""
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnP, args=(mass,num))
    theta0 = np.array([np.random.ranf(ndim) for i in range(nwalkers)])
    # theta0[:,0] = (25-5)*theta0[:,0] + 5
    # theta0[:,1] = (-1)*theta0[:,1]
    pos,prob,state = sampler.run_mcmc(theta0,100)
    sampler.reset()
    sampler.run_mcmc(pos,100)
    print('The autocorrelation time:%f\n'%emcee.autocorr.integrated_time(sampler.flatchain[:,1]))
    # return np.percentile(sampler.flatchain[:,0],50), np.percentile(sampler.flatchain[:,1],50)
    return sampler

# def mcmc(ndim,nwalkers,Mmin,mass,num):
#     theta0 = np.empty([nwalkers,ndim],dtype=float)
#     # theta0[:,0] = (17-13)*np.random.random(nwalkers) + 13      #Set initial M_max between 5M_sun and 25M_sun
#     theta0[:,0] = np.random.random(nwalkers)                   #Intercept
#     theta0[:,1] = -1*np.random.random(nwalkers) - 0.5
#     sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(mass,num))
#     sampler.run_mcmc(theta0,200)
#     print('mcmc(): The autocorrelation time:%f\n'%emcee.autocorr.integrated_time(sampler.flatchain[:,1]))
#     return sampler

print ("Generating data..")
data = random_masses(-1.35,3,15,10000)

# To fit for the PDF, fit a straight line to the histogram of the data
# in log-log space.
hist,bin_edges = np.histogram(data,bins=200)
bins = (bin_edges[:-1]+bin_edges[1:])/2

ndim,nwalkers = 2,200
sampler = mcmc(ndim,nwalkers,bins,hist)
print("Ran the Markov chain! Plotting now..")

f1,ax1 = plt.subplots(1,1)
n = ax1.hist(data,bins=100,label='Histogram of masses drawn')
x = np.linspace(3,15,100); y = x**(-1.35); ynorm = max(n[0])*y/max(y)
ax1.plot(x,ynorm,'r',label='Power law IMF given')
ax1.set_xlabel(r'Mass [$M_{\odot}$]')
ax1.set_ylabel(r'$\xi(m)$')
ax1.legend()

f2,ax2 = plt.subplots(1,1)
ax2.loglog(bins,hist,label='Histogram')
ax2.loglog(x,ynorm,label='Given')
ax2.set_xlabel(r'Mass [$M_{\odot}$]')
ax2.set_ylabel(r'$\xi(m)$')
ax2.legend()

# f2 = corner.corner(sampler.flatchain,labels=('M_max','alpha'),show_titles=True,truths=[15,1.35])

plt.show()
