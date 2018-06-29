import numpy as np

# support functions for perturbation analysis in parabolic partial differential equations
# most code obtained from github repository
# https://github.com/tclaudioe/Scientific-Computing

def cubic_spline(x, y, end=None, k1=0, k2=0, p1=0, p2=0):
    #x: x-coordinates of points
    #y: y-coordinates of points
    #end: Natural, Adjusted, Clamped, Parabolically, NaK
    
    n = len(x)
    A = np.zeros((3*n-3, 3*n-3))
    b = np.zeros(3*n-3)
    
    delta_x=np.diff(x)
       
    #Building the linear system of equations
    
    #1st property
    for i in np.arange(n-1):
        b[i]= y[i+1]-y[i]
        A[i,3*i:3*(i+1)] = [delta_x[i],delta_x[i]**2,delta_x[i]**3]
    #2nd property
    for i in np.arange(n-2):
        A[(n-1)+i,3*i:3*(i+1)+1]=[1, 2*delta_x[i], 3*delta_x[i]**2, -1]
    #3rd property
    for i in np.arange(n-2):
        A[(n-1)+(n-2)+i,3*i:3*(i+1)+2] = [0, 2, 6*delta_x[i], 0, -2]
    
    #Ending conditions (4th property)
    if end =='Natural':
        A[-2,1]= 2
        A[-1,-2] = 2
        A[-1,-1] = 6*delta_x[-1]

    elif end == 'Adjusted':
        A[-2,1]= 2
        A[-1,-2] = 2
        A[-1,-1] = 6*delta_x[-1]
        b[-2:] = [k1,k2]

    elif end == 'Clamped':
        A[-2,0]=1
        A[-1,-3:] = [1,2*delta_x[-1],3*delta_x[-1]**2]
        b[-2:] = [p1,p2]

    elif end == 'Parabolically':
        A[-2,2]=1
        A[-1,-1]=1

    elif end == 'NaK':
        A[-2,2:6]=[6,0,0,-6]
        A[-1,-4:]=[6,0,0,-6]
    
    #Solving the system
    sol = np.linalg.solve(A,b)
    S = {'b':sol[::3],
         'c':sol[1::3],
         'd':sol[2::3],
         'x':x,
         'y':y
        }
    return S

def cubic_spline_eval(xx,S):
    x=S['x']
    y=S['y']
    b=S['b']
    c=S['c']
    d=S['d']
    n=len(x)
    yy=float("nan")
    for i in np.arange(n-1):
        if x[i] <= xx and xx <= x[i+1]:
            yy = y[i]+b[i]*(xx-x[i])+c[i]*(xx-x[i])**2+d[i]*(xx-x[i])**3
    return yy

def cubic_spline_eval2(xx,S):
    x=S['x']
    y=S['y']
    b=S['b']
    c=S['c']
    d=S['d']
    n=len(x)
    yy=np.zeros_like(xx)
    for i in np.arange(n-1):
        jj = np.where(np.logical_and(x[i]<=xx,xx<=x[i+1]))
        yy[jj]=y[i]+b[i]*(xx[jj]-x[i])+c[i]*(xx[jj]-x[i])**2+d[i]*(xx[jj]-x[i])**3
    return yy
    

def bisect(f, a, b, verb=False, tol=10e-12):
    fa = f(a)
    fb = f(b)
    i = 0
    # Just checking if the sign is not negative => not root  necessarily 
    if np.sign(f(a)*f(b)) >= 0:
        print('f(a)f(b)<0 not satisfied!')
        return None
    
    if verb:
        #Printing the evolution of the computation of the root
        print(' i |     a     |     c     |     b     |     fa    |     fc     |     fb     |   b-a')
        print('----------------------------------------------------------------------------------------')
    
    while(b-a)/2 > tol:
        c = (a+b)/2.
        fc = f(c)
        if verb:
            print('%2d | %.7f | %.7f | %.7f | %.7f | %.7f | %.7f | %.7f' % (i+1, a, c, b, fa, fc, fb, b-a))
        # Did we find the root?
        if fc == 0:
            print('f(c)==0')
            break
        elif np.sign(fa*fc) < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        i += 1
        
    xc = (a+b)/2.
    return xc
    
def simpsons(myfun, N, a, b, verbose=False, text="", figname=""):
    f = np.vectorize(myfun) # So we can apply it to arrays without trouble
    x = np.linspace(a, b, N+1) # We want N bins, so N+1 points
    if N%2==1:
        if verbose: print("Simpsons rule only applicable to even number of segments")
        return np.nan
    dx = x[1]-x[0]
    xleft   = x[:-2:2]
    xmiddle = x[1::2]
    xright  = x[2::2]
    int_val = sum((f(xleft)+4*f(xmiddle)+f(xright))*dx/3)
    if verbose:
        xbin, ybin = simpsons_bins(f, xleft, xmiddle, xright)
        plot(f, xbin, ybin, int_val, N, text, figname)
    return int_val
