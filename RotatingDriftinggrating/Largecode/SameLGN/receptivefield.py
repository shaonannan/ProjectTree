import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 

conndlgnon  = np.int32(np.load('align_dlgnon.npy'))
conndlgnoff = np.int32(np.load('align_dlgnoff.npy'))

mapretino = np.zeros((12*20*6*20,1))
lgn_thresh  = 0.0
plt.figure(1)

for yv in range(2*24-6,4*24+6,18):
    for xv in range(int(2*24-2),int(4*24+6),18):
        idxy = xv + yv *12*24
        # on lgn
        idonlgn = np.squeeze(conndlgnon[idxy,np.where(conndlgnon[idxy,:])])
        mapretino[idonlgn] = 1.0
        
        # off lgn
        idofflgn = np.squeeze(conndlgnoff[idxy,np.where(conndlgnoff[idxy,:])])
        mapretino[idofflgn] = -1.0
        
    for xv in range(int(6*24-2),int(8*24+6),18):
        idxy = xv + yv *12*24
        # on lgn
        idonlgn = np.squeeze(conndlgnon[idxy,np.where(conndlgnon[idxy,:])])
        mapretino[idonlgn] = 1.0
        
        # off lgn
        idofflgn = np.squeeze(conndlgnoff[idxy,np.where(conndlgnoff[idxy,:])])
        mapretino[idofflgn] = -1.0
        
mapretino = np.reshape(mapretino,(6*20,12*20))        
plt.imshow(np.transpose(mapretino),cmap='gray')
plt.show()