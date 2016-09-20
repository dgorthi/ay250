import numpy as np
import matplotlib.pyplot as plt
from   scipy import integrate
import seaborn as sns
import emcee
import corner

sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

def random_masses(alpha, Mmin, Mmax, N):
    """
    Draw N mass values from a power law distribution with index alpha
    """
    return (Mmin-Mmax)*np.random.power(alpha,N)+ Mmax

def imf(m,alpha):
    """
    Compute the IMF given the mass range and power law index
    """
    return m**(-alpha)

def lnP(theta,mass):
    """
    Return the log likelihood of the mass, given by the IMF. theta= (Mmin,Mmax,alpha)
    """    
    Mmin,Mmax,alpha = theta[0],theta[1],theta[2]
    c,err = integrate.quad(imf,Mmin,Mmax,args=alpha)
    logP = np.sum(-alpha*np.log(mass)-np.log(c))
    return logP

def mcmc(ndim,nwalkers,**params):
    ndim =3

    if ('Mmin' in params.keys()):
        Mmin = params['Mmin']
        ndim -= 1; pos=1
    if ('Mmax' in params.keys()):
        Mmax = params['Mmax']
        ndim -= 1; pos=2
    if ('alpha' in params.keys()):
        alpha = params['alpha']
        ndim -= 1; pos=3

    theta0 = [np.random.ranf(ndim) for i in range(nwalkers)]
    if (ndim <3):
        np.insert(theta0,pos,)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=[mass])
    sampler.run_mcmc(theta0,100)



mass = random_masses(1.35,3,15,1000)


#Using emcee
ndim,nwalkers = 2,200

f1,ax1 = plt.subplots(1,1)
ax1 = plt.hist(mass, bins=40)

f2 = corner.corner(sampler.flatchain,labels=('M_min','M_max','alpha'),show_titles=True,truths=[3,15,1.35])

plt.show()
