#from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 
import os
#from connector_functions import gabor_probability
#from kernel_functions import gabor_kernel

# 12 * 10 * 6 * 10

n_pick = 8
g = 3.0

w = 1/12.0
phi = 0
gamma = 0.8  # Aspect ratio
sigma = 8
theta = 0
# >>>>>>>> Size >>>>>>>>>>>>>
def gabor_probability(sigma, Lambda, phi, gamma,theta):
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
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    psi    = phi
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb_dg   = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    clock_dg = gb_dg
    return clock_dg

Test_Gabor = gabor_probability(sigma, 1/w, 0, gamma,0)
lgn_x,lgn_y = np.shape(Test_Gabor)[0],np.shape(Test_Gabor)[1]

# Space parameters
dx = 0.6
lx = 6.0
dy = 0.6
ly = 6.0

xc = 0
yc = 0

x_values = np.arange(-lx/2, lx/2, dx)
y_values = np.arange(-ly/2, ly/2, dy)
# numbers of neurons
xnc = 1*42
ync = 1*42
nmax = xnc*ync
nlgnmax = 12

# 写之前，先检验文件是否存在，存在就删掉  
if os.path.exists("dlgnon.txt"):  
    os.remove("dlgnon.txt")  
if os.path.exists("dlgnoff.txt"):  
    os.remove("dlgnoff.txt")  
# 以写的方式打开文件，如果文件不存在，就会自动创建  
ft    = open("dlgnon.txt", 'w')  
ftoff = open("dlgnoff.txt", 'w')
align_dlgnon  = np.zeros((nmax,nlgnmax))
align_dlgnoff = np.zeros((nmax,nlgnmax))
def savetxt(filename,x):
    np.savetxt(filename,x,fmt=['%d ']*np.size(x),newline='\n')
#def gabor_probability(x, y, sigma, gamma, phi, w, theta, xc=0, yc=0):
#
#    """
#    calculate the gabor function of x and y
#
#    Returns value of the 2D Gabor function at x, y
#
#    sigma: Controls the decay of the exponential term
#    gamma: x:y proportionality factor, elongates the pattern
#    phi: Phase of the overall pattern
#    w: Frequency of the pattern
#    theta: Rotates the whole pattern by the angle theta
#    xc, yc : Linear translation
#    """
#
#    transforms_to_radians = np.pi / 180
#    theta *= transforms_to_radians
#    phi *= transforms_to_radians  # Transforms to radians
#
#    # Translate
#    x = x - xc
#    y = y - yc
#
#    # Rotate
#    aux1 = np.cos(theta) * x + np.sin(theta) * y
#    y = -np.sin(theta) * x + np.cos(theta) * y
#
#    x = aux1
#
#    # Function
#    r = x**2 + (gamma * y) ** 2
#    exp_part = np.exp(- r / (2 * sigma**2))
#    cos_part = np.cos(2 * np.pi * w * x + phi)
#
#    return exp_part * cos_part



def GaborSample(x_value,y_value,sigma,gamma,phi,w,theta,n_pick,polarity):
#    Z = np.zeros((x_values.size, y_values.size))
#    L = np.zeros(Z.shape)
#    Anat = np.zeros(Z.shape)
#    for x_index, x in enumerate(x_values):
#        for y_index, y in enumerate(y_values):
#            probability = polarity * gabor_probability(x, y, sigma, gamma, phi, w, theta, xc, yc)
#            #print('proba', probability)
#            counts = np.random.rand(n_pick) < probability
#            #print('counts', counts)
#            aux = np.sum(counts)  # Samples
#            synaptic_weight = (g / n_pick) * aux
#            L[x_index, y_index] = aux
#            Z[x_index, y_index] = synaptic_weight
#            Anat[x_index,y_index] = probability
    gb_prob = gabor_probability(sigma, 1/w, phi, gamma,theta)
    xy_shape = np.shape(gb_prob)
    Z = np.zeros((xy_shape[0], xy_shape[1]))
    L = np.zeros(Z.shape)
    for x_index in range(xy_shape[0]):
        for y_index in range(xy_shape[1]):
            probability = polarity * gb_prob[x_index,y_index]
            #print('proba', probability)
            counts = np.random.rand(n_pick) < probability
            #print('counts', counts)
            aux = np.sum(counts)  # Samples
            synaptic_weight = (g / n_pick) * aux
            L[x_index, y_index] = aux
            Z[x_index, y_index] = synaptic_weight
    return (L,Z)

countx = 0
county = 0
RFon  = {}
RFoff = {}
Bon   = {}
Boff  = {}
dlgnon = {}
dlgnoff = {}
# load theta and phase
theta  = np.zeros((1,42*42)) # np.loadtxt('theta.txt')
#theta  = theta /np.pi * 180.0
phase  = np.zeros((1,42*42)) # np.loadtxt('phase.txt')
#phase  = phase / 4.0 * 420.0
            
for yv in range(0,42,1):
    countx = 0
    for xv in range(0,42,1):
        # for lgnon/lgnoff index
        lgnon  = np.zeros((1*lgn_x,1*lgn_y))
        lgnoff = np.zeros((1*lgn_x,1*lgn_y))
        idxy   = xv + yv *1*42
        phase[0,idxy] = np.random.randint(4)*1.0
        phi    = 360*phase[0,idxy]/4/1.0 # 0#phase[idxy]
        phi    = phi/180.0*np.pi
        ##### >>>>>>>>>>>>>>>>> !!!!!!!!!!!!!!!!!!1 Change !!!!!!!!!!!!!!!!!!!
        tgv = yv - 21.5
        tgh = xv - 21.5
        the    = 0.5*np.arctan2(tgh,tgv) * 180/np.pi # theta[idxy]
        theta[0,idxy] = the*1.0/180*np.pi
        the   = theta[0,idxy]
        #print(idxy,the)
        polarity = 1 # ON
        Lon,Zon   = GaborSample(x_values,y_values,sigma,gamma,phi,w,the,n_pick,polarity)
        polarity  = -1 # OFF
        Loff,Zoff = GaborSample(x_values,y_values,sigma,gamma,phi,w,the,n_pick,polarity)
        
        Bon[idxy]   = Lon
        Boff[idxy]  = Loff
        lgnon = Bon[idxy]
        x_y         = np.where(lgnon)
        dlgnon[idxy]  = x_y[:][0]+x_y[:][1]*1*lgn_x
        lgnoff= Boff[idxy]
        x_y         = np.where(lgnoff)
        dlgnoff[idxy]  = x_y[:][0]+x_y[:][1]*1*lgn_x
        ttcount  = (county*3 + countx +1)
        
        #ft.write(str(np.reshape(dlgnon[idxy],(-1)))+'\n')
        chooseon = np.arange(np.size(dlgnon[idxy]))
        np.random.shuffle(chooseon)
        for ion in range(np.size(dlgnon[idxy])):
            ft.write(str(dlgnon[idxy][ion])+' ')
            if ion<nlgnmax:
                align_dlgnon[idxy,ion] = dlgnon[idxy][chooseon[ion]]
        ft.write('\n')
        chooseoff= np.arange(np.size(dlgnoff[idxy]))
        np.random.shuffle(chooseoff)
        for ioff in range(np.size(dlgnoff[idxy])):
            ftoff.write(str(dlgnoff[idxy][ioff])+' ')
            if ioff<nlgnmax:
                align_dlgnoff[idxy,ioff] = dlgnoff[idxy][chooseoff[ioff]]
        ftoff.write('\n')

        #plt.figure(ttcount)
        #plt.imshow(lgnon,cmap='jet')
#        plt.figure(2)
#        plt.subplot(6,12,ttcount)
#        plt.imshow(RFoff)
        countx +=1
        #print(countx,county,ttcount)
    county +=1
ft.close()      
ftoff.close()   
np.save("align_dlgnon.npy",align_dlgnon)   
np.save("align_dlgnoff.npy",align_dlgnoff)  
np.save("theta.npy",theta)
np.save("phase.npy",phase)

