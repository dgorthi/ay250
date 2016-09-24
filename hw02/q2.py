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

def lnP(alpha,Mmax,Mmin,mass):
    """
    Return the log likelihood of the mass, given by the IMF. theta= (Mmin,Mmax,alpha)
    """    
    #    Mmin,Mmax,alpha = theta[0],theta[1],theta[2]
    #    print(Mmin,Mmax,alpha)
    c,err = integrate.quad(imf,Mmin,Mmax,args=alpha)
    logP = np.sum(-alpha*np.log(mass)-np.log(c))
    return logP

def mcmc(data,nwalkers,**params):
    ndim =3
    mass =data
    if ('Mmin' in params.keys()):
        min_mass = params['Mmin']
        print(min_mass)
        ndim -= 1
        #        theta0[:,0] = min_mass
        #        arguments.append([])
    if ('Mmax' in params.keys()):
        max_mass = params['Mmax']
        ndim -= 1
        #        theta0[:,1] = max_mass
    if ('alpha' in params.keys()):
        a = params['alpha']
        ndim -= 1
        #        theta0[:,2] = a
        
    theta0 = np.array([np.random.ranf(ndim) for i in range(nwalkers)])
    print(np.shape(theta0))
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=[3,mass])
    return sampler.run_mcmc(theta0,100)

data = random_masses(1.35,3,15,1000)
print("Generated random values..")

nwalkers = 200
sampler = mcmc(data,nwalkers,Mmin=3)
print("Ran the Markov chain! Plotting now..")

f1,ax1 = plt.subplots(1,1)
ax1 = plt.hist(mass, bins=40)

f2 = corner.corner(sampler.flatchain,labels=('M_min','M_max','alpha'),show_titles=True,truths=[3,15,1.35])

plt.show()
