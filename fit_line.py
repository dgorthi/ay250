import numpy as np
import matplotlib.pyplot as plt
import emcee

def fake_data(m,b,N):
    """Generates N fake data points for the straight line of slope m and
    intercept b, with gaussian uncertainities for each data point

    """
    x = np.random.uniform(-10,10,N)
    err = np.random.normal(loc=0,scale=1,size=N)
    y = m*x + b + err
    
    f,ax = plt.subplots(1,1)
    ax.errorbar(x,y,yerr=err,fmt='o')
    plt.show()
    
    return y

