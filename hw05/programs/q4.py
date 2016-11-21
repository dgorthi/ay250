import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import fsps
import emcee
import corner

#Define constants
mab0 = -2.5*np.log10(3631e-23) # Zero pt
C = 4.56785766e-19             # B band lumin to mass
C_fuv = 43.35                  # FUV lumin to SFR

def mass(bmag,dist,N):
    """Converts a B band apparent magnitude into a stellar mass estimate
    using the distance to the galaxy. 0.1 magnitude errors are assumed
    for the B band magnitudes and these errors are forward propagated
    to the error in mass.  
    Returns: Logarithm of mass and log error in mass for all the
    galaxies in the sample.
    """
    Mb    = bmag+5-5*np.log10(dist*1e6)
    Mberr = np.absolute(np.random.normal(loc=0,scale=0.1,size=N))

    bflux = 10**-((Mb+mab0)/2.5)
    #bflux_err = np.log(10)*bflux*Mberr/2.5

    Lumin = bflux*4*np.pi*(10*3.086e18)**2
    #Lumin_err = bflux_err*4*np.pi*(10*3.086e18)**2

    mass = C*Lumin
    #mass_err = C*Lumin_err

    #logmass_err = mass_err/(np.log(10)*mass)
    logmass_err = Mberr/2.5
    return np.log10(mass),logmass_err


def sfr(fuv,fuv_err,dist,N):
    """Converts the Galex FUV filter luminosity into an estimate of the
    SFR of the galaxy. The errors in measurement of FUV magnitude are
    propagated to the errors in SFR estimated.
    Returns: Logarithm of the SFR and the log error in sfr for all the
    galaxies in the sample.
    """
    mfuv = fuv+5-5*np.log10(dist*1e6)
    flux = 10**-((mfuv+mab0)/2.5)
    flux_err = np.log(10)*flux*fuv_err/2.5
    Lfuv = flux*4*np.pi*(10*3.086e18)**2
    Lfuv_err = flux_err*4*np.pi*(10*3.086e18)**2
    fil = fsps.get_filter('galex_fuv')
    nu_eff = 2.9979e18/fil.lambda_eff
    sfr = nu_eff*Lfuv/(10**C_fuv)
    sfr_err = nu_eff*Lfuv_err/(10**C_fuv)
    logsfr_err = sfr_err/(np.log(10)*sfr)
    return np.log10(sfr),logsfr_err

def lnprior(theta):
    if (0 < theta[0] < 1) and (-9 < theta[1]< -7) :
        return theta
    return -np.inf,-np.inf

def lnP(theta,mass,mass_err,sfr,sfr_err):
    """ theta= (m,c) """
    lp = lnprior(theta)
    if (np.isfinite(lp)).all == False:
        return -np.inf

    mass_fit = np.random.normal(loc=mass,scale=np.absolute(mass_err))
    sfr_fit = theta[0]*(mass_fit) + theta[1]
    prob = norm.logpdf(sfr,loc=sfr_fit,scale=np.absolute(sfr_err))
    return np.sum(prob)
#    chi2 = 0.5*(sfr-theta[1]-theta[0]*mass)**2/(sfr_err**2 + (theta[0]*mass_err)**2)
#    return -np.sum(chi2)
    

## Take only the galaxies for which both B-band magnitudes and Galex
## FUV magnitudes are available. There seem to be some galaxies for
## which the FUV magnitude is available without an error associated
## with it. I have taken these values and assigned an error of 0.5mag
## (larger than the maximum error on this measurement for any galaxy)

dist,bmag,fuv,fuv_err = np.loadtxt('ps5.data',comments='#',usecols=(7,8,11,12)).T
mask = np.where((bmag!=99.99)&(fuv!=99.99))
N = np.size(mask)
dist, bmag, fuv, fuv_err = dist[mask], bmag[mask], fuv[mask], fuv_err[mask]

err_mask = np.where(fuv_err==99.99)
fuv_err[err_mask]= 0.5

mass,merr = mass(bmag,dist,N)
sfr ,serr = sfr(fuv,fuv_err,dist,N)

## Fit a straight line to this data with it's uncertainities.
## TODO: ACCOUNT FOR INTRINSIC SCATTER IN THE DATA

ndim,nwalkers = 2,100
theta0 = np.array([np.random.rand(ndim) for i in range(nwalkers)])
theta0[:,1] = theta0[:,1]-8
theta0[:,0] = theta0[:,0]

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(mass,merr,sfr,serr))
sampler.run_mcmc(theta0,50)

## Diagnostic plots
f1,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(sampler.chain[:,:,0])#,'.-')
ax1.set_title('Slope--walker')
ax2.plot(sampler.chain[:,:,1])#,'.-')
ax2.set_title('Intercept--Walker')

f2,(ax3,ax4)=plt.subplots(2,1)
ax3.plot(sampler.lnprobability[:,0],'.-')
ax3.set_title('Slope- probability')
ax4.plot(sampler.lnprobability[:,1],'.-')
ax4.set_title('Intercept- probability')

## Corner plot
corner.corner(sampler.flatchain,labels=('Slope','Intercept'),show_titles=True)

slope = np.percentile(sampler.flatchain[:,0],50)
intercept = np.percentile(sampler.flatchain[:,1],50)

fakemass = np.linspace(4,11,100)
fakesfr = slope*fakemass + intercept

fig,ax = plt.subplots(1,1)
ax.errorbar(mass,sfr,xerr=merr,yerr=serr,fmt='o')
ax.set_xlabel(r'$\log{Mass}$')
ax.set_ylabel(r'$\log{SFR}$')
ax.plot(fakemass,fakesfr,'r')


