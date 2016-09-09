import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

def fake_data(m,b,N):
    """Generates N fake data points for the straight line of slope m and
    intercept b, with gaussian uncertainities for each data point.

    """
    x = np.random.uniform(-10,10,N)
    err = np.random.normal(loc=0,scale=2,size=N)
    y = m*x + b + err
    
    return [x,y,err]

def lnP(theta,data):
    """Generates the log likelihood probability of the given parameters
    (m,b) fitting the data, from the inverse squared deviation of the
    fit. Assuming flat priors. Formula: P(m,b|D) = P(D|m,b)P(m|b)P(b)

    """
    err = np.sum((data[1]-(theta[1]*data[0]+theta[0]))**2)
    return (100/err)
    
def next(theta):
    """The proposal density is a normal distribution of width 5
    (determined by hit and trial for this particular problem),
    centered around theta.

    """
    return np.random.normal(loc=theta,scale=1)

def markov_chain(dat):
    """Generates a markov chain to sample a given log likelihood
    function. Calls functions lnP() and next() to compute the log
    likelihood of a given parameter set and the generate the next
    parameter set, respectively.

    """
    theta0 = [0,0]              #Starting point
    theta1 = next(theta0)
    Pcurr  = lnP(theta0,dat)

    theta,a= theta0,np.array([])
    
    for i in range(10**4):
        theta1 = next(theta0)
        Pnext  = lnP(theta1,dat)       
        a = np.append(a,min(1,np.exp(Pnext-Pcurr)))
        if a[i]==1:
            theta = np.vstack((theta,theta1))
            theta0 = theta1; Pcurr = Pnext
            continue
        elif a[i]>np.random.ranf():
            theta = np.vstack((theta,theta1))
        else:
            theta = np.vstack((theta,theta0))
    print('Scale:%d\tMean acceptance ratio:%f\n'%(4,np.mean(a)))
    return theta,a

data = fake_data(m=5,b=-2,N=100)

f,ax = plt.subplots(1,1)
ax.errorbar(data[0],data[1],yerr=data[2],fmt='o')

theta,a = markov_chain(data)
figure = corner.corner(theta)

plt.show()
#line_fit = np.dot(inv(np.dot(np.dot(A.T,inv(C)),A)),np.dot(np.dot(A.T,inv(C)),Y))


