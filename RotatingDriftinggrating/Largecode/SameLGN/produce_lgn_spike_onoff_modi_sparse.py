import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 

# >>> Input Pattern >>>
import Inpututilities_beta01 as inp

def utilhvs(x):
    if x>=0:
        return x
    else:
        return 0
def utilhvsvec(x):
    f = np.maximum(x,0)
    return f


(dt,tfinal) = (1e-3,0.6)
omega        = 200.0/180.0* np.pi/1000.0 # 4Hz
sigma,gamma  = 8, 0.8
Lambda       = 12
ft           = 40.0/1000.0 * np.pi
dg  = inp.drifting_spatio_grating(sigma, Lambda, 0, gamma,omega,ft,dt,1,tfinal*1000)

# >>>>>>>> Scale of LGN-layer 12 * 20 * 6 * 20 (1) >>>>>
xlgn = 1*19
ylgn = 1*19
nlgn = xlgn*ylgn



glontemporal  = np.zeros((nlgn,int(tfinal/dt)))
glofftemporal = np.zeros((nlgn,int(tfinal/dt)))
tau_e = 0.002
treal_onset = 0.000#min(ton_onset,toff_onset)

#plt.figure(1)
#plt.plot(fr_brt_off)
#plt.plot(fr_drk_on)
#plt.show()
counter = 0
bright_relate_amp = 1.0
for itt in range(int(tfinal/dt)):
    tt = itt*dt   
    dg_on  = np.reshape(dg[itt],-1)
    idx_on = np.where(dg_on>0)
    
    glontemporal[idx_on,itt] = dg_on[idx_on]
    
    dg_off = np.reshape(dg[itt],-1)
    idx_off= np.where(dg_off<0)
    glofftemporal[idx_off,itt] = -dg_off[idx_off]
    

conndlgnon  = np.int32(np.load('align_dlgnon.npy'))
conndlgnoff = np.int32(np.load('align_dlgnoff.npy'))

gl = np.zeros((1*42*1*42,int(tfinal/dt)))
lgn_thresh  = 0.0
# plt.figure()

sparse_d = 0.725
sparse_b = 0.725#sparse_d/5.0 * 2.0
for yv in range(0,1*42,1):
    for xv in range(0,42,1):
        idxy = xv + yv *1*42
        # on lgn
        idonlgn = np.squeeze(conndlgnon[idxy,np.where(conndlgnon[idxy,:])])
        if np.size(idonlgn) <= 1:
            continue
        ratio_amp = 18.0/len(idonlgn)
        gl[idxy,:] += np.sum(glontemporal[idonlgn,:],axis = 0) * ratio_amp
        
        # off lgn
        idofflgn = np.squeeze(conndlgnoff[idxy,np.where(conndlgnoff[idxy,:])])
        if np.size(idofflgn) <= 1:
            continue
        ratio_amp = 18.0/len(idofflgn)
        gl[idxy,:] += 1.0 * np.sum(glofftemporal[idofflgn,:],axis = 0)* ratio_amp

        npshow = np.squeeze(gl[idxy,:])
        gl[idxy,:] = utilhvsvec(npshow) * 2.0 #/ sparse_d
        sparse_index = np.random.random(1)
#        if sparse_index > sparse_d:
#            gl[idxy,:] = 0
            
        gl[idxy,:]  = utilhvsvec(gl[idxy,:] - 20)
        
        """
        """
        plt.subplot(2,1,1)
        plt.plot(gl[idxy,:])
        plt.ylim([0,80])
#for yv in range(0,42,1):
#    for xv in range(18,1*42,1):
#        idxy = xv + yv *1*42
#        # on lgn
#        idonlgn = np.squeeze(conndlgnon[idxy,np.where(conndlgnon[idxy,:])])
#        gl[idxy,:] += np.sum(glontemporal[idonlgn,:],axis = 0)
#        
#        # off lgn
#        idofflgn = np.squeeze(conndlgnoff[idxy,np.where(conndlgnoff[idxy,:])])
#        gl[idxy,:] += 1.0 * np.sum(glofftemporal[idofflgn,:],axis = 0)
#
#        npshow = np.squeeze(gl[idxy,:])
#        gl[idxy,:] = utilhvsvec(npshow) * 4.0 #/ sparse_d
#        sparse_index = np.random.random(1)
##        if sparse_index > sparse_d:
##            gl[idxy,:] = 0
#            
#        gl[idxy,:]  = utilhvsvec(gl[idxy,:] - 20)
#        
#        """
#        """
#        plt.subplot(2,1,2)
#        plt.plot(gl[idxy,:])
#        plt.ylim([0,80])
                

plt.show()
theta  = (np.load('theta.npy'))
phase  = (np.load('phase.npy'))
scio.savemat('pythondata.mat', {'gl':gl,'phase':phase,'theta':theta})  


"""
stimulus= np.zeros((12*42,6*42))
stimulus[int(2.5*42):int(4.5*42),42*2:4*42] = 0.5
stimulus[int(6.5*42):int(8.5*42),42*2:4*42] = 1.0
plt.imshow(stimulus,cmap="gray")
"""
