import numpy as np
from   scipy import special
from   scipy import integrate
import matplotlib.pyplot as plt
import emcee
import corner
## Compute the SFR density using observed UV luminosity functions.
## Parameters for z<1 taken from Arnouts et al (2005)
## 
## Parameters for 2<z<4 taken from Reddy et al (2009)
## Parameters for z>4 taken from Bouwens et al (2015)

C_fuv = 43.35                    # SFR to log(FUV_lumin)
mab0  = -2.5*np.log10(3631e-23)  # AB Mag zero pt
pccm  = 3.086e18                 # pc->cm conversion
c     = 2.99792458e18            # Ang/s

z     = np.array([0.055,  0.3,  0.5,  0.7,  1.0, 1.14, 1.75, 2.23,  2.3, 3.05,  3.8,  4.9,  5.9,  6.8,  7.9, 10.4])
lambd = np.array([ 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1700, 1700, 1600, 1600, 1600, 1600, 1600, 1600])
nueff = c/lambd

Mstar = np.array([-18.05, -18.38, -19.49, -19.84, -20.11, -19.62, -20.24, -19.87,  -20.70, -20.97, -20.88, -21.17, -20.94, -20.87, -20.63, -20.92])
Merr  = np.array([  0.11,   0.25,   0.37,   0.40,   0.45,   0.06,   0.32,   0.18,    0.11,   0.14,   0.08,   0.12,   0.20,   0.26,   0.36,   0.00])
#Convert magnitude to luminosity
Lstar = (10**-((Mstar+mab0)/2.5))*4*np.pi*(10*pccm)**2
Lerr  = (np.log(10)/2.5)*Lstar*Merr

phistar = np.array([4.07, 6.15, 1.69, 1.67, 1.14, 2.96, 3.11, 3.32, 2.75, 1.71, 1.97, 0.74, 0.50, 0.29, 0.21, 0.008])*1e-3
phierr  = np.array([0.56, 1.76, 0.88, 0.95, 0.76, 0.15, 1.61, 0.91, 0.54, 0.53, 0.32, 0.16, 0.17, 0.16, 0.17, 0.003])*1e-3

alpha = np.array([-1.21, -1.19, -1.55, -1.60, -1.63, -1.48, -1.48, -1.48, -1.73, -1.73, -1.64, -1.76, -1.87, -2.06, -2.02, -2.27])
aerr  = np.array([ 0.07,  0.15,  0.21,  0.26,  0.45,  0.62,  0.62,  0.62,  0.07,  0.13,  0.04,  0.05,  0.10,  0.13,  0.23,  0.00])

def phi(L,alpha):
    return (L**(alpha+1))*np.exp(-L)

fuv,err = np.zeros(np.size(z),dtype=np.float64),np.zeros(np.size(z))

for i in range(np.size(z)):
    fuv[i],err[i] = integrate.quad(phi,0.001,np.inf,args=(alpha[i]))

# Check if you got it right
assert(np.isclose(fuv[0],special.gamma(2+alpha[0])*special.gammaincc(2+alpha[0],0.001)))

logsfr = np.log10(nueff*fuv*phistar*Lstar)-C_fuv

## Next, estimate errors. The maximum value of the integral is
## obtained for (1) Steepest alpha (most negative) (2) Largest L* and
## (3) Largest phi*

fuvmax,fuvmin = np.zeros(np.size(z),dtype=np.float64), np.zeros(np.size(z),dtype=np.float64)

for i in range(np.size(z)):
    fuvmax[i] = integrate.quad(phi,0.001,np.inf,args=(alpha[i]-aerr[i]))[0]
    fuvmin[i] = integrate.quad(phi,0.001,np.inf,args=(alpha[i]+aerr[i]))[0]

fuverr = fuvmax-fuvmin

sfrerr = np.sqrt((phierr/phistar)**2 + (Lerr/Lstar)**2 + (fuverr/fuv)**2)

## Now fit functional form of eq(7) from Finkelstein to this SFH using
## the errors computed.

def lnprior(theta):
    if (0<theta[0]<5) & (2<theta[1]<4) & (0<theta[2]<10) & (0<theta[3]<10):
        return 0.0
    return -np.inf

def lnP(theta,sfh,sfherr,z):
    """Log likelihood of parameters theta=(A,B,\alpha,\gamma) given the
    SFH, the errors and z
    """
    lp = lnprior(theta)
    if not(np.isfinite(lp)):
        return -np.inf    
    A,B,alpha,gamma = theta
    #    sfh_pred = A*(1+z)**alpha/(1+((1+z)/B)**gamma)
    sfh_pred = np.log10(A*(1+z)**alpha/(1+((1+z)/B)**gamma))
    chi2 = ((sfh_pred-sfh)/0.3)**2

    return -np.sum(chi2/2)

ndim,nwalkers=4,300
#sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(10**(logsfr),np.log(10)*logsfr*sfrerr,z))
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(logsfr,sfrerr,z))


theta0 = np.array([np.random.rand(ndim) for i in range(nwalkers)])
# Start with the best fit values from Madau and Dickinson (eq 15)
theta0[:,0] = (0.02-0.01)*theta0[:,0]+0.01
theta0[:,1] = (4-2)*theta0[:,1]+2
theta0[:,2] = (3-2)*theta0[:,2]+2
theta0[:,3] = (6-5)*theta0[:,3]+5

# Burn-in for a bit
pos, prob, state = sampler.run_mcmc(theta0,1000)
sampler.reset()

sampler.run_mcmc(pos,1000)

## Diagnostic plots
f1,(ax1,ax2,ax3,ax4)=plt.subplots(4,1)
ax1.plot(sampler.chain[:,:,0])
ax1.set_title(r'$A$')
ax2.plot(sampler.chain[:,:,1])
ax2.set_title(r'$B$')
ax3.plot(sampler.chain[:,:,2])
ax3.set_title(r'$\alpha$')
ax4.plot(sampler.chain[:,:,3])
ax4.set_title(r'$\gamma$')

f2,ax5=plt.subplots(1,1)
ax5.plot(sampler.lnprobability,'.-')
ax5.set_title(r'lnP')

for i in range(ndim):
    plt.figure()
    plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
    plt.title("Dimension {0:d}".format(i))

## Corner plot
corner.corner(sampler.flatchain,labels=(r'$A$',r'$B$',r'$\alpha$',r'$\gamma$'),show_titles=True)#,title_fmt='.2e')

A = np.percentile(sampler.flatchain[:,0],50)
B = np.percentile(sampler.flatchain[:,1],50)
a = np.percentile(sampler.flatchain[:,2],50)
g = np.percentile(sampler.flatchain[:,3],50)
sfh_theo = A*(1+z)**a/(1+((1+z)/B)**g)
sfh_md = 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)

## Plot result of fit
f,ax = plt.subplots(1,1)
ax.errorbar(z[:5], logsfr[:5], yerr=sfrerr[:5], fmt='s',label='Arnouts+05')
ax.errorbar(z[5:8],logsfr[5:8],yerr=sfrerr[5:8],fmt='o',label='Dahlen+07')
ax.errorbar(z[8:10],logsfr[8:10],yerr=sfrerr[8:10],fmt='D',label='Reddy+09')
ax.errorbar(z[10:], logsfr[10:], yerr=sfrerr[10:], fmt='p',label='Bouwens+15')
ax.plot(z,np.log10(sfh_theo),label='Best Fit')
ax.plot(z,np.log10(sfh_md),label='MD14')
ax.legend()
