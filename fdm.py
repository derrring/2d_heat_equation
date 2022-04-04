import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Stix2"],
})



def discretization(L, t_MAX, h=0.05, k=0.001 ):
    # L: length of the rod
    # t_MAX: maximum time
    # h: discretization of the rod
    # k: discretization of the time
    rod = np.arange(0, L+h ,h)
    t_interval = np.arange(0, t_MAX+k, k)

    rod_knots = len(rod)  
    time_knots = len(t_interval)

    return rod, t_interval, rod_knots, time_knots, h, k, L, t_MAX

def heat_source(x, t):
    f = np.exp**(-t**2)*np.sin(x) #define at each time
    # f= 0
    return f

def initial_condition(x):
    return np.sqrt(x)

def func_bc(t):
    return np.sin(t/t[-1]*np.pi)


def euler_explicit(rod, t_interval, rod_knots, time_knots, h, k):
    u = np.zeros(( rod_knots,time_knots))
    #BC:
    u[0, 1: ]= func_bc(t_interval[1:])
    u[-1,1:] = func_bc(t_interval[1:])
    #IC:
    u[: ,0]= initial_condition(rod)

    #A_h: quadratic differential matrix
    A = np.zeros((rod_knots-2,rod_knots-2 ))
    np.fill_diagonal(A, 2)
    A += (np.eye(rod_knots-2,rod_knots-2,k=1) + np.eye(rod_knots-2,rod_knots-2,k=-1))
    A_h = 1/(h**2)*A

    for j in range(time_knots[1:-1]):
        u[:,j+1] = np.matmul( (np.eye(rod_knots-2,rod_knots-2)-k*A_h), u[:,j]) + k*heat_source( rod[1:-1], t_interval[j])
    return u 

def euler_implicit(rod, t_interval, rod_knots, time_knots, h, k):
    pass

def leapforg(rod, t_interval, rod_knots, time_knots, h, k):
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
    for j in range(1,time_knots-1):##
        u[:,j+1] =u[:,j-1] -np.matmul( (2*k*A_h), u[:,j]) + 2*k*heat_source( rod[0:-1], t_interval[j-1])
    return u 



'''
Crank-Nicolson method
'''
def matrix_CF(N, func):
    # N as the number of knots (INCLUDING END POINTS)
    h = 1/(N-1)
    h_coeff = h**(-2)

    # Creation of diagonal part of C_h
    vect0 = h_coeff*2 * np.ones(N)
    C0 = np.diagflat(vect0, 0)

    # upper part of C_h
    vect1 = h_coeff*(-1+h) * np.ones(N-1)
    C1 = np.diagflat(vect1, 1)

    # lower part of C_h
    vect2 = h_coeff*(-1-h)*np.ones(N-1)
    C2 = np.diagflat(vect2, -1)

    # assembly C_h
    C_h = C0 + C1 + C2

    x = np.linspace(0, 1, N)
    F_h = func(x)

    return C_h, F_h

def crank_nicolson(N, M, f):
    h = 1/(N-1)
    k = 1/(M-1)
    t = np.linspace(0, 1, M)
    x = np.linspace(0, 1, N)
    U = np.zeros((N, M+1))
    # U[:, 0] = u(x,0) = 0 already satisfied

    C_h, F_h = matrix_CF(N, f)

    U[:, 0] = N*[0]
    U[0, :] = U[-1, :] = (M+1)*[0]
    for j in range(M):
        U[:, j+1] = np.linalg.solve(np.eye(N)+k/2*C_h, 
                np.matmul((np.eye(N) - k/2*C_h), U[:, j]) + k*F_h)

    return t, U

if __name__ == '__main__' :
    [ rod, t_interval, rod_knots, time_knots, h, k, L, t_MAX] = discretization(5, 100)
    u =  euler_explicit(rod, t_interval, rod_knots, time_knots, h, k)
    fig =plt.figure()
    ax2= fig.add_subplot(projection="3d")

    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$t$")
    ax2.set_zlabel(r"$u(x,t)$")
    ax2.set_ylim(t_MAX,0)

    ax2.plot_surface(rod,t_interval, u, cmap="rainbow")
    fig.show()