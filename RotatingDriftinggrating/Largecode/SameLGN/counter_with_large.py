# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Python script file for Simoncelli 2002 Motion illusions as oPtimal percepts. 
which could be further used in quantitative measurement for Line -motion illusion
"""
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from ipywidgets import interact
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from io import StringIO



def imshow(win,p):
    """ show result figure in 2D format(x--spatial, t--temporal)."""
    win.imshow(p.T,orgin='lower',interpolation='nearest',cmap='gray')
    xmid = (p.shape[1]-1) / 2
    ymid = (p.shape[0]-1) / 2
    win.vlines(ymid,0,p.shape[0],color='gray')
    win.hlines(xmid,0,p.shape[1],color='gray')
    win.set_xticks([])
    win.set_yticks([])
    
def data_Preprocess():
    fvm = np.loadtxt('vmem.txt')
    fvm = np.reshape(fvm,[250,41472])
    """
    np.reshape, extract data and then, at first list in line, when current line is fullfilled,
    turn to the next line
    """
    return fvm
def Conv2d_Gaussian(orgImg,sigma):
    """ original gradient map and given sigma """
    h = gaussian_filter(orgImg,sigma = sigma,mode='constant',cval=0.0)
    return h

def uniform(x,low,high):
    """ compute log probability for a variable with uniform distribution([low,high])"""
    return scipy.stats.uniform.logpdf(x,low,high-low)

def norm(x,mu,sigma):
    """ simmilar to 'uniform',compute log probability for a variable with normal distribution"""
    return scipy.stats.norm.logpdf(x,mu,sigma)

"""
calculating differential matrix for dvdx(spatial) and dvdt(temporal)
"""
def FirstOrder_deviation(fz,dz):
    [m,n] = np.size(fz)
    vxl   = np.diff(fz[0:-1,:],axis=0)
    vxr   = np.diff(fz[1:,:],axis=0)
    dvdx  = 0.5 * vxl[:,1:-1] + 0.5 * vxr[:,1:-1]
    
    vtl   = np.diff(fz[:,0:-1],axis=1)
    vtr   = np.diff(fz[:,1:],axis =1)
    dvdt  = 0.5 * vtl[1:-1,:] + 0.5 * vtr[1:-1,:]
    
    return (dvdx,dvdt)

"""
ScaleParam = [24,12,6,0.5,250]
model(ScaleParam)
"""
def model(ScaleParam):
    nc    = ScaleParam[0]
    nxhyp = ScaleParam[1]
    nyhyp = ScaleParam[2]
    xh    = ScaleParam[3]
    
    ntt   = ScaleParam[4]
    
    dxx   = xh / nc
    dtt   = 1.0
    c_stm   = 60
    bin_len = 8
    """
    narray configuration
    """    
    dvdx = np.zeros([nxhyp*nc-2,ntt-2])
    dvdt = np.zeros_like(dvdx)
    
    fdis = np.zeros([nc*nxhyp,nc*nyhyp,ntt])
    fsmt = np.zeros_like(fdis)
    """ start to do pre-process """
    fvm = data_Preprocess()
    for i in range(250):
        fztemp = np.reshape(np.squeeze(fvm[i,:]),[nc*nyhyp,nc*nxhyp])
        fdis[:,:,i] = np.transpose(fztemp,[1,0])
        fsmt[:,:,i] = Conv2d_Gaussian(np.squeeze(fdis[:,:,i]),28)
    return fsmt

ScaleParam = [24,12,6,0.5,250]
nc    = ScaleParam[0]
nxhyp = ScaleParam[1]
nyhyp = ScaleParam[2]
xh    = ScaleParam[3]

ntt   = ScaleParam[4]

dxx   = xh / nc
dtt   = 1.0
c_stm   = 60
bin_len = 8

"""￼
narray configuration
"""    
dvdx = np.zeros([nxhyp*nc-2,ntt-2])
dvdt = np.zeros_like(dvdx)

fdis = np.zeros([nc*nxhyp,nc*nyhyp,ntt])
fsmt = np.zeros_like(fdis)
fz   = np.zeros([nc*nxhyp,ntt])
""" start to do pre-process """
fvm = data_Preprocess()
fvlarge = np.zeros((nc*nxhyp,8*nc*nyhyp))
for i in range(250):
    fztemp = np.reshape(np.squeeze(fvm[i,:]),[nc*nyhyp,nc*nxhyp])
    fdis[:,:,i] = np.transpose(fztemp,[1,0])
    fsmt[:,:,i] = Conv2d_Gaussian(np.squeeze(fdis[:,:,i]),16)
    fz[:,i]     = np.mean(np.squeeze(fsmt[:,(c_stm-bin_len+1):(c_stm+bin_len+1),i]),axis=1)
    
counter = 0
for i in range(33,110,10):    
    fvlarge[:,counter*nc*nyhyp:(counter+1)*nc*nyhyp] = np.mean(fsmt[:,:,i-2:i+3],axis=2)
    counter +=1
"""
fvlarge[:,0:(0+1)*nc*nyhyp] = fsmt[:,:,35]
counter = 1
for i in range(50,65,10):
    fvlarge[:,counter*nc*nyhyp:(counter+1)*nc*nyhyp] = fsmt[:,:,i]
    #if i==50:
        #fvlarge[:,counter*nc*nyhyp:(counter+1)*nc*nyhyp] /= 1.50
        
    counter +=1   
for i in range(75,95,10):
    fvlarge[:,counter*nc*nyhyp:(counter+1)*nc*nyhyp] = 1.0*fsmt[:,:,i]#1.05*fsmt[:,:,i]
    counter +=1  
"""
fz = Conv2d_Gaussian(fz,12)
fig = plt.figure()
plt.imshow(fvlarge/1.0,cmap='jet',vmin=0.0,vmax=0.65)

fig  = plt.figure()
plt.xlabel('Time [ms]',fontsize = 18)
plt.ylabel('Cortical distance [mm]',fontsize = 18)
#plt.title('Propagation activity',fontsize = 32)
plt.title('model-LGN inputs',fontsize = 32)
#plt.title('NMDA-type input',fontsize = 32)

plt.xticks([72,72+144,72+144*2,72+144*3,72+144*4],['40','50','60','70','80'],fontsize = 16)

plt.yticks([0,144,287],['0.0','3.0','6.0'],fontsize = 16)

prop = plt.imshow(fvlarge,cmap="jet",vmin = 0.0,vmax = 0.8)

plt.figure()
plt.imshow(fz[:,:200]*1.0,cmap='jet')
plt.clim([0,1.40])
"""
plt.figure(6)
plt.title('Propagation activity',fontsize = 32)
"""
fsup = fzof+fzon -fzb-fzc
plt.figure()
plt.imshow(fsup[:,:200]*1.0,cmap='jet')
plt.clim([0,0.30])   


fssup = fzsof+fzson -fzsb-fzsc
plt.figure()
plt.imshow(fssup[:,:200]*1.0,cmap='jet')
plt.clim([0,0.30])   