import numpy as np
import matplotlib.pyplot as plt
import fsps

dist,bmag = np.loadtxt('ps5.data',comments='#',usecols=(7,8)).T
# add 0.1mag uncertainities to the b-magnitudes
M = bmag+5-5*np.log10(dist*1e6)
Merr = np.random.normal(loc=0,scale=0.1,size=np.size(dist))
Mb = M+Merr

# z = (6.27+\-0.21) + (-0.11+\-0.01)M_B 

# Z = np.random.normal(loc=6.27,scale=0.21,size=np.size(dist))
# +(np.random.normal(loc=-0.11,scale=0.01,size=np.size(dist)))*Mb
# +np.random.normal(loc=0,scale=0.15,size=np.size(dist))
# Zerr = Z-(6.27-0.11*Mb)

#Z = 6.27-0.11*Mb
Z = np.random.normal(loc=6.27,scale=0.21,size=np.size(dist)) + (np.random.normal(loc=-0.11,scale=0.01,size=np.size(dist)))*Mb
Zerr = np.random.normal(loc=0,scale=0.15,size=np.size(dist))
Zf = Z+Zerr

fig,ax = plt.subplots(1,1)
ax.errorbar(Mb,Zf,xerr=Merr,yerr=Zerr,fmt='o')
#ax.plot(Mb,Z,'o')
ax.set_xlabel(r'$M_B$')
ax.set_ylabel(r'$12+\log{O/H}$')
ax.invert_xaxis()

plt.show()
