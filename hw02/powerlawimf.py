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

def lnprior(theta,Mmin):
    """Imposing the known priors on theta=(Mmax,alpha). Mmax>Mmin for
    integral to exist and alpha should be negative to keep the
    function normalizable

    """
    Mmax,alpha = theta[0],theta[1]
    if (-5 < alpha < 0) and (Mmax>Mmin):
        return 0
    return -np.inf

def lnP(theta,mass,num):
    """The maximum likelihood function for fitting a power law is same as
    that for fitting a line in its log-log space.

    """
    lp = lnprior(theta,3)
    if not np.isfinite(lp):
        return -np.inf
    Mmax,alpha = theta[0],theta[1]
    imf = lambda m,alpha: m**alpha
    c,err = integrate.quad(imf,3,Mmax,args=alpha)
    if c<=0:
        print("lnP(): c less than zero encountered")
        return -np.inf
    n_pred = alpha*mass-np.log(c)
    d = norm.logpdf(num,loc=n_pred,scale=0.01)
    return np.sum(d)

def mcmc(ndim,nwalkers,mass,num):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnP, args=(mass,num))
    theta0 = np.array([np.random.ranf(ndim) for i in range(nwalkers)])
    theta0[:,0] = (25-5)*theta0[:,0] + 5
    theta0[:,1] = (-1)*theta0[:,1]
    pos,prob,state = sampler.run_mcmc(theta0,100)
    sampler.reset()
    sampler.run_mcmc(pos,100)
    print('The autocorrelation time:%f\n'%emcee.autocorr.integrated_time(sampler.flatchain[:,1]))
    # return np.percentile(sampler.flatchain[:,0],50), np.percentile(sampler.flatchain[:,1],50)
    return sampler

print ("Generating data..")
data = random_masses(-1.35,3,15,10)

# To fit for the PDF, fit a straight line to the histogram of the data
# in log-log space. MORAL: To fit for the right intercept the area
# needs to be normalized under the histogram.
hist,bin_edges = np.histogram(data,bins=3)
bins = (bin_edges[:-1]+bin_edges[1:])/2
hist = hist/integrate.trapz(hist,bins)

ndim,nwalkers = 2,300
sampler = mcmc(ndim,nwalkers,np.log(bins),np.log(hist))
print("Ran the Markov chain! Plotting now..")

# f1,ax1 = plt.subplots(1,1)
# n = ax1.hist(data,bins=200,label='Histogram of masses drawn')
# x = np.linspace(3,15,100); y = x**(-1.35); ynorm = max(n[0])*y/max(y)
# ax1.plot(x,ynorm,'r',label='Power law IMF given')
# ax1.set_xlabel(r'Mass [$M_{\odot}$]')
# ax1.set_ylabel(r'$\xi(m)$')
# ax1.legend()

# f2,ax2 = plt.subplots(1,1)
# ax2.loglog(bins,hist,label='Histogram')
# ax2.loglog(x,ynorm,label='Given')
# ax2.set_xlabel(r'Mass [$M_{\odot}$]')
# ax2.set_ylabel(r'$\xi(m)$')
# ax2.legend()

f2 = corner.corner(sampler.flatchain,labels=('Mmax','alpha'),show_titles=True)

plt.show()
