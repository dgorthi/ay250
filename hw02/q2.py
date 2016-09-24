import numpy as np
import matplotlib.pyplot as plt
from   scipy import integrate
import seaborn as sns
import emcee
import corner

sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

def random_masses(alpha, Mmin, Mmax, N):
    """Draw N mass values from a power law distribution with index alpha"""
    return (Mmin-Mmax)*np.random.power(alpha,N)+ Mmax

def imf(m,alpha):
    """Compute the IMF given the mass range and power law index"""
    return m**(-alpha)

def lnprior(theta):
    """Imposing the known priors on parameters theta= (Mmin,Mmax,alpha)"""
    Mmin,Mmax,alpha = theta[0],theta[1],theta[2]
    if 1 < alpha < 2:
        Mmin = 3
        return Mmin, Mmax, alpha
    return -np.inf,-np.inf,-np.inf
    
def lnP(theta,mass):
    """Return the log likelihood of the mass, given by the IMF. theta=
    (Mmin,Mmax,alpha)"""
    lp = lnprior(theta)
    if (np.isfinite(lp)).all == False:
        return -np.inf
    Mmin, Mmax, alpha = lp[0],lp[1],lp[2] 
    c,err = integrate.quad(imf,Mmin,Mmax,args=alpha)
    if c<0:
        return -np.inf
    logP = np.sum(-alpha*np.log(mass)-np.log(c))
    if np.isfinite(logP):
        return logP
    return -np.inf

def mcmc(ndim,nwalkers,mass):
    # ndim =3
    # if ('Mmin' in params.keys()):
    #     min_mass = params['Mmin']
    #     print(min_mass)
    #     ndim -= 1
    #     #        theta0[:,0] = min_mass
    #     #        arguments.append([])
    # if ('Mmax' in params.keys()):
    #     max_mass = params['Mmax']
    #     ndim -= 1
    #     #        theta0[:,1] = max_mass
    # if ('alpha' in params.keys()):
    #     a = params['alpha']
    #     ndim -= 1
    #     #        theta0[:,2] = a
        
    theta0 = np.array([np.random.ranf(ndim) for i in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=([mass]))
    sampler.run_mcmc(theta0,100)
    return sampler

data = random_masses(1.35,3,15,1000)
print("Generated random values..")

ndim,nwalkers = 3,200
sampler = mcmc(ndim,nwalkers,data)
print("Ran the Markov chain! Plotting now..")

f1,ax1 = plt.subplots(1,1)
ax1 = plt.hist(data,bins=50)

#f2 = corner.corner(sampler.flatchain,labels=('M_min','M_max','alpha'),show_titles=True,truths=[3,15,1.35])

plt.show()
