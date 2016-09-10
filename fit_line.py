## Program to generate fake data with m=5, b=-2 and then to fit this
## fake data using a MCMC sampler implemented using the
## Metropolis-Hastings algorithm.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee
import corner

sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

def fake_data(m,b,N):
    """Generates N fake data points for the straight line of slope m and
    intercept b, with gaussian uncertainities for each data point.

    """
    x = np.random.uniform(-10,10,N)
    err = np.random.normal(loc=0,scale=3,size=N)
    y = m*x + b + err
    
    return x,y,err

def lnP(theta,x,y):
    """Generates the log likelihood probability of the given parameters
    (m,b) fitting the data, from the inverse squared deviation of the
    fit. Assuming flat priors. Formula: P(m,b|D) = P(D|m,b)P(m|b)P(b)

    """
    err = np.sum((y-(theta[1]*x+theta[0]))**2)
    return (1/err)
    
def next(theta):
    """The proposal density is a normal distribution of width 5
    (determined by hit and trial for this particular problem),
    centered around theta.

    """
    return np.random.normal(loc=theta,scale=[2,1])

def markov_chain(x,y):
    """Generates a markov chain to sample a given log likelihood
    function. Calls functions lnP() and next() to compute the log
    likelihood of a given parameter set and the generate the next
    parameter set, respectively.

    """
    theta0 = [0,0]              #Starting point
    theta1 = next(theta0)
    Pcurr  = lnP(theta0,x,y)

    theta,a= theta0,np.array([])
    
    for i in range(10**4):
        theta1 = next(theta0)
        Pnext  = lnP(theta1,x,y)       
        a = np.append(a,min(1,Pnext/Pcurr))
        if a[i]==1:
            theta = np.vstack((theta,theta1))
            theta0 = theta1; Pcurr = Pnext
            continue
        elif a[i]>np.random.ranf():
            theta = np.vstack((theta,theta1))
        else:
            theta = np.vstack((theta,theta0))
    print('Mean acceptance ratio:%f\n'%np.mean(a))
    return theta[1:],a

x,y,err = fake_data(m=5,b=-2,N=100)
theta,a = markov_chain(x,y)

f1,ax = plt.subplots(1,1)
ax.errorbar(x,y,yerr=err,fmt='o')
#ax.plot((theta[0]+theta[1]*x))
ax.set_xlabel('x')
ax.set_ylabel('y')

f2 = corner.corner(theta,bins=50,labels=('intercept','slope'),plot_contours=True,top_ticks=True,truths=[-2.,5.])

plt.show()
#line_fit = np.dot(inv(np.dot(np.dot(A.T,inv(C)),A)),np.dot(np.dot(A.T,inv(C)),Y))


