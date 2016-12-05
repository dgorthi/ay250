import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# -------------------------------------------------------------------------#
#                                  DATA                                    #
# -------------------------------------------------------------------------#
#The magnitudes are absolute magnitudes in the AB system.

mab0 = -2.5*np.log10(3631e-23) # Zero pt

#-------------------------------- z = 4 -----------------------------------#
Mz4 = np.linspace(-22.69,-17.69,endpoint=True,num=11)
Mz4 = np.append(Mz4,[-16.94,-15.94])
phiz4 = np.array([3, 15, 134, 393, 678, 1696, 2475, 2984, 5352, 6865,
                  10473, 24580, 25080])*1e-6
errz4 = np.array([4,  9,  23,  40,  63,  113,  185,  255,  446, 1043,
                   2229,  3500,  7860])*1e-6

#------------------------------- z = 5 ------------------------------------#
Mz5 = np.linspace(-23.11,-19.11,endpoint=True,num=9)
Mz5 = np.append(Mz5,[-18.36,-17.36,-16.36])
phiz5 = np.array([2, 6, 34, 101, 265, 676, 1029, 1329, 2085, 4460,
                  8600, 24400])*1e-6
errz5 = np.array([2, 3,  8,  14,  25,  46,   67,   94,  171,  540,
                  1760,  7160])*1e-6

#------------------------------- z = 6 ------------------------------------#
Mz6 = np.linspace(-22.52,-19.52,endpoint=True,num=7)
Mz6 = np.append(Mz6,[-18.77,-17.77,-16.77])
phiz6 = np.array([2, 15, 53, 176, 320, 698, 1246, 1900, 6680, 13640])*1e-6
errz6 = np.array([2,  6, 12,  25,  41,  83,  137,  320, 1380,  4200])*1e-6

#------------------------------- z = 7 ------------------------------------#
Mz7 = np.linspace(-22.66,-18.66,endpoint=True,num=9)
Mz7 = np.append(Mz7,[-17.91,-16.91])
phiz7 = np.array([0,  1, 33, 48, 193, 309, 654, 907, 1717, 5840, 8500])*1e-6
errz7 = np.array([2,  2,  9, 15,  34,  61, 100, 177,  478, 1460, 2946])*1e-6

#------------------------------- z = 8 -----------------------------------#
Mz8 = np.linspace(-22.87,-19.37,endpoint=True,num=8)
Mz8 = np.append(Mz8,[-18.62,-17.62])
phiz8 = np.array([0, 0, 5, 13, 58, 60, 331, 533, 1060, 2740])*1e-6
errz8 = np.array([2, 2, 3,  5, 15, 26, 104, 226,  340, 1040])*1e-6

#------------------------------- z = 10 ----------------------------------#
Mz10 = np.linspace(-22.23,-18.23,endpoint=True,num=5)
phiz10 = np.array([0, 1, 10,  0, 266])*1e-6
errz10 = np.array([1, 1,  5, 49, 171])*1e-6

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

fig,ax= plt.subplots(1,1)
ax.set_yscale("log", nonposy='clip')
ax.errorbar(Mz4, phiz4, yerr=errz4, fmt='o-',label=r'$z=4$' )
ax.errorbar(Mz5, phiz5, yerr=errz5, fmt='o-',label=r'$z=5$' )
ax.errorbar(Mz6, phiz6, yerr=errz6, fmt='o-',label=r'$z=6$' )
ax.errorbar(Mz7, phiz7, yerr=errz7, fmt='o-',label=r'$z=7$' )
ax.errorbar(Mz8, phiz8, yerr=errz8, fmt='o-',label=r'$z=8$' )
#ax.errorbar(Mz10,phiz10,yerr=errz10,fmt='o-',label=r'$z=10$')
ax.invert_xaxis()
ax.legend()
    
def lnprior(theta):
    if (1e-5<theta[0]<1e-1) & (-22<theta[1]<-15) & (-2<theta[2]<0):
        return 0.0
    return -np.inf

def lnP(theta,M,phi,err):
    """ theta=(phi*, M*, alpha)"""
    lp = lnprior(theta)
    if not(np.isfinite(lp)):
        return -np.inf    
    phi_pred = 0.4*np.log(10)*theta[0]*np.power(10,0.4*(theta[2]+1)*(theta[1]-M))*np.exp(-np.power(10,0.4*(theta[1]-M)))
    chi2 = np.sum(((phi-phi_pred)/err)**2)/2
    return -chi2

# def lnprior(theta):
#     if (1e-7<theta[0]<1e-1) & (1e27<theta[1]<1e30) & (-3<theta[2]<0):
#         return 0.0
#     return -np.inf

# def lnP(theta,L,phi,err):
#     """ theta=(phi*,L*,alpha) """

#     lp = lnprior(theta)
#     if not(np.isfinite(lp)):
#         return -np.inf
#     phi_pred = theta[0]*((L/theta[1])**theta[2])*np.exp(-L/theta[1])
#     chi2 = np.sum(((phi-phi_pred)/err)**2)/2
#     return -chi2
    
# Lz4 = (10**-((Mz4+mab0)/2.5))*4*np.pi*(10*3.086e18)**2

ndim,nwalkers=3,300
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(Mz4,phiz4,errz4))

theta0 = np.array([np.random.rand(ndim) for i in range(nwalkers)])
theta0[:,0] = (phiz4.max()-phiz4.min())*theta0[:,1]+phiz4.min()
theta0[:,1] = (22-15)*theta0[:,1]-22
theta0[:,2] = 2*theta0[:,2]-2

# Burn-in for a bit
pos, prob, state = sampler.run_mcmc(theta0,1500)
sampler.reset()

sampler.run_mcmc(pos,500)

## Diagnostic plots
f1,(ax1,ax2,ax3)=plt.subplots(3,1)
ax1.plot(sampler.chain[:,:,0])
ax1.set_title(r'$\phi^*$')
ax2.plot(sampler.chain[:,:,1])
ax2.set_title(r'$L_*$')
ax3.plot(sampler.chain[:,:,2])
ax3.set_title(r'$\alpha$')

f2,ax4=plt.subplots(1,1)
ax4.plot(sampler.lnprobability,'.-')
ax4.set_title(r'lnP')

for i in range(ndim):
    plt.figure()
    plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
    plt.title("Dimension {0:d}".format(i))

## Corner plot
corner.corner(sampler.flatchain,labels=(r'$\phi^*$',r'$M_*$',r'$\alpha$'),show_titles=True)#,title_fmt='.2e')

## Final parameters
phi_star = np.percentile(sampler.flatchain[:,0],50)
M_star = np.percentile(sampler.flatchain[:,1],50)
alpha = np.percentile(sampler.flatchain[:,2],50)

phi_pred = (np.log(10)/2.5)*phi_star*np.power(10,0.4*(M_star-Mz4)*(alpha+1))*np.exp(-np.power(10,0.4*(M_star-Mz4)))
# phi_pred = (phi_star/L_star)*((Lz4/L_star)**alpha)*np.exp(-(Lz4/L_star))
f3,ax5 = plt.subplots(1,1)
ax5.set_yscale("log", nonposy='clip')
# ax5.set_xscale("log", nonposy='clip')
ax5.plot(Mz4,phi_pred,'o-',label='Best Fit')
ax5.errorbar(Mz4,phiz4,yerr=errz4,fmt='o-',label='Data')
ax5.legend()

# M_star = -2.5*np.log10(L_star/(4*np.pi*(10*3.086e18)**2))-mab0
print("Final values for z=4: phi* = %.2e    alpha = %.2f    M*=%.2f"%(phi_star,alpha,M_star))

