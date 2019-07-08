from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence_velo_pres(error_velo,error_pres,list_N):
    # velocity convergence plot
    fig = plt.figure()
    axis = fig.gca()
    linear=error_velo[::-1]
    axis.loglog(list_N,linear,label='$||e_u||_{\infty}$')
    axis.loglog(list_N,0.0001*np.power(list_N,2),'r*',label="second order")
    axis.loglog(list_N,0.0001*np.power(list_N,1),'g*',label="first order")
    axis.set_xlabel('$2**Level$')
    axis.set_ylabel('$Error$')
    axis.legend()
    plt.show()

    #pressure convergence plot
    fig = plt.figure()
    axis = fig.gca()
    linear=error_pres[::-1]
    axis.loglog(list_N,linear,label='$||e_p||_{\infty}$')
    axis.loglog(list_N,0.1*np.power(list_N,2),'r*',label="second order")
    axis.loglog(list_N,0.1*np.power(list_N,1),'g*',label="first order")
    axis.set_xlabel('$2**Level$')
    axis.set_ylabel('$Error$')
    axis.legend()
    plt.show()