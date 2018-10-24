import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# '''
# First Step!
# Generating Gabor-filter Bank
# '''
# sigma,gamma = 8, 0.8
# Lambda      = 12
# GaborBank_yx = MomentGabor(4,4,sigma,gamma,Lambda)
def drifting_spatio_grating(sigma, Lambda, psi, gamma,omega,ft,dt,dtl,tfinal):
    NPHA = 4
    '''
    ft: time frequency
    lambda: space frequency
    sigma: Bounding box
    '''
    ntt      = int(tfinal/dtl)
    ntt_real = int(tfinal/dt)
    bin_nt   = int(dtl/dt)

    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    nstds = 1.2 # Number of standard deviation sigma
    thetao = np.pi/4.0
    xmax = max(abs(nstds * sigma_x * np.cos(thetao)), abs(nstds * sigma_y * np.sin(thetao)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(thetao)), abs(nstds * sigma_y * np.cos(thetao)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
#    (xmax,xmin,ymax,ymin) = (13,-13,13,-13)
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    nmax = 4#16#36#
    
    clock_dg      = {}

#    omega  = 20/180.0*np.pi/1000

    for itorien in range(ntt):
        '''
        omega-theta : orientation
        ft - phase : phase
        '''
        t_curr = dtl * itorien
        theta  = t_curr * omega
        phase  = t_curr * ft
        psi    = phase
        

        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb_dg   = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
        clock_dg[itorien] = gb_dg


    return clock_dg



