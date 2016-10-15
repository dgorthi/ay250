import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats import norm
import fsps
import emcee
import corner

#zmet =1 => [Fe/H] -2.50
#zmet =2 => [Fe/H] -2.00

def gen_data(filters,sigma):

    print('Initializing Stellar Population..')
    ssp = fsps.StellarPopulation(zmet=2,dust_type=1)
    
    print('Computing magnitude in the V and I bands..')
    V,I = ssp.get_mags(tage=14,bands=f)
    
    mags = [V,I]
    mags+= -5 + 5*np.log10(8.2e3) + np.random.normal(loc=0,scale=sigma,size=2)
    return mags

def lnprior(theta):
    """theta = (zmet,age,distance)"""
    z,age,dist = theta[0],theta[1],theta[2]
    if 8 < dist < 10:
        if 5 < age < 14:
            if (-2.50 <= z < -2.25):
                return 1,age,dist
            elif (-2.25 <= z <= -2):
                return 2,age,dist
            else:
                return -np.inf, -np.inf, -np.inf
        else:
            return -np.inf, -np.inf, -np.inf
    else:
        return -np.inf, -np.inf, -np.inf

def lnP(theta,data,ssp,sigma):
    """theta = (zmet,age,distance)"""

    lp = lnprior(theta)
    print (lp)
    if (np.isfinite(lp)).all() == False:
        return -np.inf
    z,age,dist = lp[0],lp[1],lp[2] 

    mag = ssp.get_mags(zmet=z,tage=age,bands=f)
    mag+= -5 + 5*np.log10(dist*1e3)
    
    d = norm.logpdf(data,loc=mag,scale=sigma)
    return np.sum(d)


f = [fsps.find_filter('f606w')[1],fsps.find_filter('f814w')[1]]
#data = gen_data(f,0.1)
data = [ 20.58183853,  20.37690413]

ssp = fsps.StellarPopulation(dust_type=1)
ndim,nwalkers = 3,100
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(data,ssp,0.1))

theta0 = np.array([np.random.ranf(ndim) for i in range(nwalkers)])
theta0[:,0] = (-2.5+2)*theta0[:,0]-2.00
theta0[:,1] = (14-5)*theta0[:,1]+5
theta0[:,2] = (10-8)*theta0[:,2]+8

sampler.run_mcmc(theta0,100)

print("Ran the Markov chain! Plotting now..")

corner.corner(sampler.flatchain,labels=('z','age','distance'),show_titles=True)
