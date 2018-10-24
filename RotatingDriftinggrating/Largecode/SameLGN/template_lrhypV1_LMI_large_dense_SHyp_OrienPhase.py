import numpy as np
import matplotlib.pyplot as plt
import Inpututilities_beta01 as inp

(dt,tfinal) = (1e-3,0.2)

omega        = 20.0/180.0* np.pi/1000.0 # 4Hz
sigma,gamma  = 8, 0.8
Lambda       = 12
ft           = 4.0/1000.0 * np.pi
dg  = inp.drifting_spatio_grating(sigma, Lambda, 0, gamma,omega,ft,1,dt,tfinal)







