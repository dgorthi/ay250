import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from   matplotlib import rc
import matplotlib

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=3)

x = np.linspace(1,10,100)
y1 = (11-x)**3
y1norm = y1/max(y1)
y2 = x**-3
y2norm = y2/max(y2)

plt.plot(x,y1norm,label=r'$(11-x)^3$')
plt.plot(x,y2norm,label=r'$x^{-3}$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()
