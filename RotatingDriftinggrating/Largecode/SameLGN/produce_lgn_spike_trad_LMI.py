#from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 
# 12 * 20 * 6 * 20 nlgn
xlgn = 12*20
ylgn = 6*20
nlgn = xlgn*ylgn

(t1on,t2on,t1off,t2off) = (0.014,0.056,0.014/0.056*0.036,0.036)#(0.014/0.056*0.036,0.036,0.014/0.056*0.036,0.036)#
(dt,tfinal) = (1e-3,0.2)
t_pulse     = 0.020
def temporalkernel(t1,t2,dt,tfinal):
    tt = np.arange(0,tfinal,dt)
    f  = tt/(t1**2)*np.exp(-tt/t1) - tt/(t2**2)*np.exp(-tt/t2)
    return f

def utilhvs(x):
    if x>=0:
        return x
    else:
        return 0
def utilhvsvec(x):
    f = np.maximum(x,0)
    return f
    
def real_RGC_input(t_pulse,dt,tfinal,t1on,t1off,t2on,t2off):    
    n_pulse= int(t_pulse/dt)
    t_pulse = np.ones(n_pulse)/n_pulse/1.0*8.0
    ton  = temporalkernel(t1on,t2on,dt,tfinal)
    toff = temporalkernel(t1off,t2off,dt,tfinal)
    t_on  = np.convolve(ton,t_pulse*dt,mode='full')
    t_off = np.convolve(toff,t_pulse*dt,mode='full')

    return 1.0*t_on*1e3,1.0*t_off*1e3
t_pulse = 0.050#t_pulse = 0.050
fr_on_p,fr_off = real_RGC_input(t_pulse,dt,tfinal,t1on,t1off,t2on,t2off)
t_pulse = 0.050#t_pulse = 0.100
fr_on,fr_off_p = real_RGC_input(t_pulse,dt,tfinal,t1on,t1off,t2on,t2off)

fr_on  -= fr_off_p * 0.80
fr_off -= fr_on_p  * 0.80
#fr_on = utilhvsvec(fr_on)
#fr_off = utilhvsvec(fr_off)
# location receive stimuli
spon  = np.zeros((xlgn,ylgn))
spoff = np.zeros((xlgn,ylgn))
ton_onset  = 0.035#0.040 + 0.025
toff_onset = 0.025#0.040
"""
upper dark and bottom bright
"""
"""
spoff[50-5:90+5,2*20-5:4*20+5]  = toff_onset
spon[130-10:170+5,2*20-5:4*20+5]   = ton_onset
"""

spoff[20-5:80+5,2*20-10:4*20+10]  = toff_onset
spon[100-5:220+5,2*20-10:4*20+10]   = ton_onset

"""
plan B
spoff[55-5:95+5,2*20-5:4*20+5]  = toff_onset
spon[135-10:175+5,2*20-5:4*20+5]   = ton_onset
"""
#spon[5*20-5:10*20+5,int(2*20)-5:int(4*20)+5]   = ton_onset

"""
transpose
"""
#plt.figure(2)
#plt.subplot(1,2,1)
#plt.imshow(spon)
#plt.subplot(1,2,2)
#plt.imshow(spoff)
#plt.show()
spon  = np.reshape(np.transpose(spon),(-1))
spoff = np.reshape(np.transpose(spoff),(-1))
ideffon  = np.where(spon)
ideffoff = np.where(spoff)
#print('on: ',ideffon)
#print('off: ',ideffoff)


# generate spike trains
trefractory = 0.0015
cond0       = 30.0
pspon       = -2*trefractory*np.ones(nlgn)
pspoff      = -2*trefractory*np.ones(nlgn)
glon  = np.zeros(nlgn)
gloff = np.zeros(nlgn)
slon  = np.zeros_like(glon)
sloff = np.zeros_like(gloff)

glonp  = np.zeros(nlgn)
gloffp = np.zeros(nlgn)
slonp  = np.zeros_like(glon)
sloffp = np.zeros_like(gloff)

glonn  = np.zeros(nlgn)
gloffn = np.zeros(nlgn)
slonn  = np.zeros_like(glon)
sloffn = np.zeros_like(gloff)

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
for itt in range(int(treal_onset/dt),int(tfinal/dt),1):
    tt = itt*dt
    treal_on = utilhvs(tt - ton_onset - treal_onset)
    if treal_on<=0:
        counter +=1
        print('count',counter)
        continue
    # fake exponential decay for all
    #decay
    te  = dt/tau_e
    ete = np.exp(-te)
    glonn  = (glonp  + slonp*te) * ete
    slonn  = (slonp            ) * ete 

    randon  = np.random.rand(nlgn)
    fronmat = np.zeros_like(randon)
    fronmat[ideffon] = bright_relate_amp*fr_on[int(treal_on/dt)]
    idfron  = np.where(randon<fronmat*dt)
    
    glon[idfron] = glonp[idfron]
    slon[idfron] = slonp[idfron]
    
    # refractory time
    fakepsp = np.ones_like(pspon)
    refracpsp = np.ones_like(pspon)
    fakepsp[idfron] = tt + randon[idfron]/fronmat[idfron]
    refracpsp[idfron] = pspon[idfron] + trefractory
    ideffrefrac     = np.where(fakepsp>refracpsp)
    ideffrefrac     = np.intersect1d(ideffrefrac,idfron)
    
    idfron = ideffrefrac
    pspon[idfron]   = tt + randon[idfron]/fronmat[idfron]
    
    #rise        
    dtt = randon[idfron]/fronmat[idfron]    
    te  = dtt/tau_e
#    if(itt == 500):
#        print('firing rate: ',fronmat[idfron])
#        print('te: ',te)
    ete = np.exp(-te)
    glon[idfron]  = (glon[idfron]  + slon[idfron]*te) * ete
    slon[idfron]  = (slon[idfron]          ) * ete + cond0
    #decay
    te  = (dt-dtt)/tau_e
    ete = np.exp(-te)
    glon[idfron]  = (glon[idfron]  + slon[idfron]*te) * ete
    slon[idfron]  = (slon[idfron]          ) * ete 

        
    #update all
    glonp = glonn
    slonp = slonn

    
    glonp[idfron] = glon[idfron]
    slonp[idfron] = slon[idfron]
    
    glontemporal[:,itt] = glonp[:]
    
    
dark_relate_amp = 1.30
for itt in range(int(treal_onset/dt),int(tfinal/dt),1):
    tt = itt*dt
    treal_off = utilhvs(tt - toff_onset - treal_onset)
    if treal_off<=0:
        
        continue
    # fake exponential decay for all
    #decay
    te  = dt/tau_e
    ete = np.exp(-te)
    gloffn  = (gloffp  + sloffp*te) * ete
    sloffn  = (sloffp            ) * ete 

    randoff  = np.random.rand(nlgn)
    froffmat = np.zeros_like(randoff)
    froffmat[ideffoff] = dark_relate_amp * 1.0*fr_off[int(treal_off/dt)] #- 0.5*fr_on[int(treal_off/dt)]  # add inhibitory effect
    idfroff  = np.where(randoff<froffmat*dt)
    
    gloff[idfroff] = gloffp[idfroff]
    sloff[idfroff] = sloffp[idfroff]
    
    # refractory time
    fakepsp = np.ones_like(pspoff)
    refracpsp = np.ones_like(pspoff)
    fakepsp[idfroff] = tt + randoff[idfroff]/froffmat[idfroff]
    refracpsp[idfroff] = pspoff[idfroff] + trefractory
    ideffrefrac     = np.where(fakepsp>refracpsp)
    ideffrefrac     = np.intersect1d(ideffrefrac,idfroff)
    
    idfroff = ideffrefrac
    pspoff[idfroff]   = tt + randoff[idfroff]/froffmat[idfroff]
    
    #rise        
    dtt = randoff[idfroff]/froffmat[idfroff]    
    te  = dtt/tau_e
#    if(itt == 500):
#        print('firing rate: ',froffmat[idfroff])
#        print('te: ',te)
    ete = np.exp(-te)
    gloff[idfroff]  = (gloff[idfroff]  + sloff[idfroff]*te) * ete
    sloff[idfroff]  = (sloff[idfroff]          ) * ete + cond0
    #decay
    te  = (dt-dtt)/tau_e
    ete = np.exp(-te)
    gloff[idfroff]  = (gloff[idfroff]  + sloff[idfroff]*te) * ete
    sloff[idfroff]  = (sloff[idfroff]          ) * ete 

        
    #update all
    gloffp = gloffn
    sloffp = sloffn

    
    gloffp[idfroff] = gloff[idfroff]
    sloffp[idfroff] = sloff[idfroff]
    
    glofftemporal[:,itt] = gloffp[:]
   

for index,i in enumerate(ideffon[0]):
    index = int(index)
    i     = int(i)


#plt.figure(2)
for index,i in enumerate(ideffoff[0]):
    index = int(index)
    i     = int(i)

conndlgnon  = np.int32(np.load('align_dlgnon.npy'))
conndlgnoff = np.int32(np.load('align_dlgnoff.npy'))

gl = np.zeros((12*24*6*24,int(tfinal/dt)))
lgn_thresh  = 0.0
#plt.figure()

sparse_d = 0.725
sparse_b = 0.365#sparse_d/5.0 * 2.0
for yv in range(2*24-6,4*24+6,1):
    for xv in range(int(1.5*24-2),int(3.5*24+6),1):
        idxy = xv + yv *12*24
        # on lgn
        idonlgn = np.squeeze(conndlgnon[idxy,np.where(conndlgnon[idxy,:])])
        gl[idxy,:] += np.sum(glontemporal[idonlgn,:],axis = 0)
        
        # off lgn
        idofflgn = np.squeeze(conndlgnoff[idxy,np.where(conndlgnoff[idxy,:])])
        gl[idxy,:] += 1.0 * np.sum(glofftemporal[idofflgn,:],axis = 0)

        npshow = np.squeeze(gl[idxy,:])
        gl[idxy,:] = utilhvsvec(npshow) * 6.0 / sparse_d
        sparse_index = np.random.random(1)
        if sparse_index > sparse_d:
            gl[idxy,:] = 0
        
        """
        plt.subplot(2,1,1)
        plt.plot(gl[idxy,25:176])
        plt.ylim([0,400])
        """
    for xv in range(int(5.5*24-10),int(10.5*24+2),1): #upper dark and bottom bright
        idxy = xv + yv *12*24
        # on lgn
        idonlgn = np.squeeze(conndlgnon[idxy,np.where(conndlgnon[idxy,:])])
        gl[idxy,:] += np.sum(glontemporal[idonlgn,:],axis = 0)

        # off lgn
        idofflgn = np.squeeze(conndlgnoff[idxy,np.where(conndlgnoff[idxy,:])])
        gl[idxy,:] += np.sum(glofftemporal[idofflgn,:],axis = 0)
        npshow = np.squeeze(gl[idxy,:])
        gl[idxy,:] = utilhvsvec(npshow) * 6.0 / sparse_b
        sparse_index = np.random.random(1)
        if sparse_index > sparse_b:
            gl[idxy,:] = 0
        
        """
        plt.subplot(2,1,2)
        plt.plot(gl[idxy,25:176])
        plt.ylim([0,400])
        """

plt.show()
scio.savemat('pythondata.mat', {'gl':gl})  




"""
stimulus= np.zeros((12*24,6*24))
stimulus[int(2.5*24):int(4.5*24),24*2:4*24] = 0.5
stimulus[int(6.5*24):int(8.5*24),24*2:4*24] = 1.0
plt.imshow(stimulus,cmap="gray")
"""
