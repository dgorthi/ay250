## Draw samples from a log normal distribution and try to fit a power
## law to it. You can drawn from any given distribution using emcee.
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import matplotlib
import emcee
import corner

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

def lnprior(m,mmin,mmax):
    """Set the range in which to drawn the masses.

    """
    if mmin <= m <= mmax:
        return 0.0
    return -np.inf

def ln_imf_chabrier(m,mmin,mmax):
    """The distribution which the sampler has to draw samples from-- 
    the Chabrier IMF.

    """
    prior = lnprior(m,mmin,mmax)
    if np.isfinite(prior) == False:
        return -np.inf
    xi = prior-(np.log(m/0.08)/(2*0.69))**2 + np.log(0.15/m)
    return xi

def gen_mass(N,mmin,mmax):
    """Use emcee to draw samples from the given lognormal Chabrier IMF

    """
    nwalkers,ndim = 1000,1
    m0 = [(mmax-mmin)*np.random.ranf(ndim)+mmin for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_imf_chabrier,args=(mmin,mmax))
    #Make the sampler feel the Burn! TODO: Check if not restricting
    #the range during sampling but filtering later will decrease the
    #autocorrelation time. It's currently high with this method.
    pos,prob,state = sampler.run_mcmc(m0,100)
    sampler.reset()
    sampler.run_mcmc(pos,10)
    samples = sampler.flatchain
    print('The autocorrelation time:%f\n'%emcee.autocorr.integrated_time(samples))
    hist,bin_edges = np.histogram(samples,bins=100)
    bins = (bin_edges[:-1] + bin_edges[1:])/2
    return bins,hist

def imf(theta,mass,num):
    """Data is in log-space making alpha the slope of the fit."""
    num_pred = mass*theta[1] + theta[0]
    return -np.sum((num_pred- num)**2)

def mcmc(ndim,nwalkers,mass,num):
    """Use emcee to fit a line to the Cabrier IMF data in log space."""
    logm,logn = np.log(mass), np.log(num)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, imf, args=(logm,logn))
    theta0 = [np.random.ranf(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(theta0,1000)
    c,alpha = np.percentile(sampler.chain[:,:,0],50), np.percentile(sampler.chain[:,:,1],50)
    print ("alpha from mcmc optimization is %f"%alpha)
    return c,alpha

mass,num = gen_mass(10000,0.5,0.8)

theta_mcmc = mcmc(2,100,mass,num)
num_mcmc = theta_mcmc[0] + np.log(mass)*theta_mcmc[1]

f,ax = plt.subplots(1,1)
ax.plot(np.log(mass),np.log(num),'o',label='Generated Data')
ax.plot(np.log(mass),num_mcmc,label='emcee fit')
ax.legend()

plt.show()
