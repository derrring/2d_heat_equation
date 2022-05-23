import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Stix2"],
})
import os
from pathlib import Path
figure_output_dir =  'figures' 
if not os.path.exists(figure_output_dir):
    os.makedirs(figure_output_dir)

def discretization(L, t_MAX, h=0.5, k=0.1):
    # L: length of the rod
    # t_MAX: maximum time
    # h: discretization of the rod
    # k: discretization of the time
    rod = np.arange(0, L+h ,h)
    t_interval = np.arange(0, t_MAX+k, k)

    rod_knots = len(rod)  
    time_knots = len(t_interval)

    return rod, t_interval, rod_knots, time_knots, h, k, L, t_MAX

def heat_source(rod, time):
    #f = np.outer(np.sin(x),np.exp(-t**2)) #define at each time
    f= np.zeros((len(rod),len(time)))
    return f

def initial_condition(rod):
    ic = np.sin(rod/rod[-1]*np.pi)*100+0.5
    return ic

def func_bc(t_interval):
    bc= np.zeros(len(t_interval))
    return bc

def compability_check(rod, t_interval):
    # IC BC compability check
    try:
        #BC: 
        bc = func_bc(t_interval)
        #IC:
        ic = initial_condition(rod)
        assert(ic[0] != bc[0] or ic[-1] !=bc[0])
    except AssertionError: 
        print('IC and BC are not compatible')
    else:
        u = np.zeros(( rod_knots,time_knots))
        u[0, 1: ]=u[-1,1:] = bc[1:]
        u[: ,0]= ic
        return u, ic, bc


def euler_explicit(rod, t_interval, rod_knots, time_knots, h, k):
    u, ic, bc = compability_check(rod, t_interval)
    
    A = np.zeros((rod_knots-2, rod_knots))

    A[:,1:-1] += (2*np.eye(rod_knots-2, rod_knots-2 ,k=0)  - np.eye(rod_knots-2,rod_knots-2, k=1)  - np.eye(rod_knots-2,rod_knots-2,k=-1))
    A[:,0] = A[:,-1] = np.zeros((rod_knots-2)) 
    A_h = 1/(h**2)*A

    for j in range(time_knots-1): 
        A[0,0] = A[-1,-1] =  bc[j]
        u[1:-1,j+1] = -np.matmul( k*A_h, u[:,j]) + k*heat_source(rod[1:-1],t_interval)[:,j] + u[1:-1,j]
        u[0,j+1] = u[-1,j+1] = bc[j+1]
    return u 


def euler_implicit(rod, t_interval, rod_knots, time_knots, h, k):
    u, ic, bc = compability_check(rod, t_interval)

    A = np.zeros((rod_knots-2, rod_knots))
    A[:,1:-1] += (2*np.eye(rod_knots-2, rod_knots-2 ,k=0)  - np.eye(rod_knots-2,rod_knots-2, k=1)  - np.eye(rod_knots-2,rod_knots-2,k=-1))
    A[:,0] = A[:,-1] = np.zeros((rod_knots-2)) 
    A[0,0] = A[-1,-1] =  initial_condition(rod)[0]
    A_h = 1/(h**2)*A
    A_h_prime = A_h[:,1:-1]

    for j in range( time_knots-1):
        u[1:-1,j+1] =np.matmul( np.linalg.inv(np.eye(rod_knots-2,rod_knots-2) +  k*A_h_prime) ,  (u[1:-1,j] + k*heat_source(rod[1:-1],t_interval)[:,j+1]))
        u[0,j+1] = u[-1,j+1] = bc[j+1]
    return u


def leap_forg(rod, t_interval, rod_knots, time_knots, h, k):
    u = np.zeros(( rod_knots,time_knots))##
    #BC:
    u[0, : ]= func_bc(t_interval[0:])##
    u[-1,:] = func_bc(t_interval[0:])##
    #IC:
    u[: ,0]= initial_condition(rod)
    #A_h: quadratic differential matrix

    A = np.zeros((rod_knots,rod_knots ))
    A_h = 1/(h**2)*A
    np.fill_diagonal(A_h, 2)
    A += (np.eye(rod_knots,rod_knots,k=1) + np.eye(rod_knots,rod_knots,k=-1))

    u[:,1] = np.matmul( (np.eye(rod_knots,rod_knots)-k*A_h), u[:,0]) + k*heat_source( rod[0:-1], t_interval[-1])
    for j in range(0,time_knots-1):##
        u[:,j+1] =u[:,j-1] -np.matmul( (2*k*A_h), u[:,j]) + 2*k*heat_source( rod[0:-1], t_interval[j-1])
    return u 

'''
Crank-Nicolson method
'''
def matrix_CF(rod_knots):
    h = 1/(rod_knots-1)
    h_coeff = h**(-2)

    # Creation of diagonal part of C_h
    vect0 = h_coeff*2 * np.ones(rod_knots)
    C0 = np.diagflat(vect0, 0)

    # upper part of C_h
    vect1 = h_coeff*(-1+h) * np.ones(rod_knots-1)
    C1 = np.diagflat(vect1, 1)

    # lower part of C_h
    vect2 = h_coeff*(-1-h)*np.ones(rod_knots-1)
    C2 = np.diagflat(vect2, -1)

    # assembly C_h
    C_h = C0 + C1 + C2

    return C_h

def crank_nicolson(rod, t_interval, rod_knots, time_knots, h, k):
    u, ic, bc = compability_check(rod, t_interval)
    C_h= matrix_CF(rod_knots)

    u[:, 0] = rod_knots*[0]
    u[0, :] = u[-1, :] = (t_interval+1)*[0]
    for j in range(time_knots-1):
        u[:, j+1] = np.linalg.solve(np.eye(rod_knots)+k/2*C_h, 
                np.matmul((np.eye(rod_knots) - k/2*C_h), u[:, j]) + k*heat_source(rod, t_interval[j]))
    return u







if __name__ == '__main__' :
    [ rod, t_interval, rod_knots, time_knots, h, k, L, t_MAX] = discretization(5, 10)
    u =  euler_implicit(rod, t_interval, rod_knots, time_knots, h, k)
    fig1 =plt.figure()
    ax11= fig1.add_subplot(121, projection="3d")
    X,Y=np.meshgrid(rod,t_interval)
    ax11.set_xlabel(r"$x$")
    ax11.set_ylabel(r"$t$")
    ax11.set_zlabel(r"$u(x,t)$")
    #ax11.set_ylim(t_MAX,0)
    ax11.plot_surface(X,Y, u.T, cmap="rainbow")
    ax11.view_init(elev=20,  azim=45 )
    ax12= fig1.add_subplot(122, projection="3d")
    X,Y=np.meshgrid(rod,t_interval)
    ax12.set_xlabel(r"$x$")
    ax12.set_ylabel(r"$t$")
    ax12.set_zlabel(r"$u(x,t)$")
    #ax12.set_ylim(t_MAX,0)
    ax12.plot_surface(X,Y, u.T, cmap="rainbow")
    #plt.savefig(f'./{figure_output_dir}/fem-output-1d.png', dpi=150, format='png')
    fig1.show()

    #==========================================================

    #Algorithm analysis - consistency
    import random
    t_section = random.randint(1,time_knots-1)
    fig2 = plt.figure()
    ax21 = fig2.add_subplot(121)
    for h in [1, 0.75, 0.5, 0.2, 0.1]:
        [ rod, t_interval, rod_knots, time_knots, h, k, L, t_MAX] = discretization(5, 10, h)
        u= euler_explicit(rod, t_interval, rod_knots, time_knots, h, k)
        ax21.plot(rod, u[:,t_section] ,label=f'h={h}', linewidth=0.5, marker=',')
    ax21.set_title( f'the {t_section}th iteration with k={k}',pad = 12)
    ax21.set_xlabel(r"$x$")
    ax21.set_ylabel(r"$u(x,t)$")
    ax21.legend(loc='best')

    ax22 = fig2.add_subplot(122)
    for h in [1, 0.75, 0.5, 0.2, 0.1]:
        [ rod, t_interval, rod_knots, time_knots, h, k, L, t_MAX] = discretization(5, 10, h)
        u= euler_implicit(rod, t_interval, rod_knots, time_knots, h, k)
        ax22.plot(rod, u[:,t_section] ,label=f'h={h}', linewidth=0.5, marker=',')
    ax22.set_title( f'the {t_section}th iteration with k={k}',pad = 12)
    ax22.set_xlabel(r"$x$")
    ax22.legend(loc='best')
    #plt.savefig(f'./{figure_output_dir}/stability-consistency.png', dpi=150, format='png')

    fig2.show()