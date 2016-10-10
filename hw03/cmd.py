import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import rc
import matplotlib
import seaborn as sns
import emcee
import corner

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

# data[0] == color (R-I), data[1] == magnitude (I band)
data = np.loadtxt('NGC6341.M92/final/REAL/NGC6341R.RDVIQ.cal.adj.zpt',usecols=(5,7),skiprows=1)

f,ax = plt.subplots()
ax.plot(data[:,0],data[:,1],'.')
ax.set_xlabel(r'F606W-F814W')
ax.set_ylabel(r'F814W')
ax.set_title('NGC 6341')

plt.show()
