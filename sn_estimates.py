import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from itertools import combinations
import pandas as pd
    
## General info ##
nbins = 10
lmax = 1000
fsky = 0.36 

## NOISES ##
dat = np.loadtxt('../Data/cl_n0.txt')
ell = np.arange(len(dat))[2:lmax+1]
dl = 0.25*(ell*(ell+1))**2
MCN0_withoutFG = dat[:,1][2:lmax+1]
MCN0_withFG = dat[:,2][2:lmax+1]

## 'old' estimate of the noise: Lensing noise level: 3uK-arcmin white noise, 
## 30 arcmin FWHM gaussian beam, fullsky ##
n_kk = np.loadtxt('../Data/nlkk.dat', usecols = [1])[2:lmax+1]
noise_nofg = (dl*MCN0_withoutFG)#[2:lmax+1]
noise_fg = (dl*MCN0_withFG)#[2:lmax+1]
noises = [n_kk, noise_nofg, noise_fg]

n_gg = 2.8e-9/nbins

## DATA READING ##
filename = '../Data/lite_euclid_camb_bins_step_nl5_lmax1000_ScalarCovCls.txt'

## Power spectra: total survey##
filename_tot = '../Data/lite_euclid_camb_tot_nl5_lmax1000_ScalarCovCls.txt'
cl_x_tot = np.loadtxt(filename_tot, usecols = [12]
            )*np.pi/np.sqrt(ell*(ell+1))         
cl_p_tot = np.loadtxt(filename_tot, dtype = float, unpack = True, 
            usecols = [11])*np.pi/2
cl_g_tot = np.loadtxt(filename_tot, dtype = float,
              unpack = True, usecols = [16])*2*np.pi/(ell*(ell+1))

## S/N from total survey ##
sn_tot_survey = [(fsky*(2*ell+1)*(cl_x_tot)**2)/((cl_x_tot)**2 + (cl_g_tot + 
                 n_gg*nbins)*(cl_p_tot+noises[i])) for i in range(len(noises))]

## PLOT ##
names = ['old', 'MCN0_withoutFG', 'MCN0_withFG']
colors = ['green', 'orange', 'violet']
fig, ax = plt.subplots()
for i in range(len(noises)):
    ax.plot(ell, sn_tot_survey[i], label = names[i], lw = 1.8, color = colors[i])
    print('S/N total case:', names[i], f'= {np.sqrt(sn_tot_survey[i].sum()):.1f}')
ax.set_xscale('log', base = 10)
ax.set_xlim(2,1000)
ax.set_xlabel('$\ell$')    
ax.set_ylabel('$(S/N)^2$') 
ax.set_ylim(0,16)   

## Power spectra: tomographic case ##
name_x = ['PxW{}'.format(i) for i in range(1,nbins+1)]
name_a = ['W{}'.format(i)+'xW{}'.format(i) for i in range(1,nbins+1)]
name_gigj = []
index = np.asarray(sorted((set(combinations(np.arange(1, nbins+1,1), r =2)))), 
                          dtype = int)       
for i in range(len(index)):
    name_gigj.append('W'+str(index[i][0])+'xW'+str(index[i][1]))
    
data = pd.read_csv(filename, delim_whitespace=True, index_col=0)
cl_x = (np.asarray(data[name_x].transpose()))*np.pi/np.sqrt(ell*(ell+1))
cl_g = np.asarray(data[name_a].transpose())*2*np.pi/(ell*(ell+1))
cl_p = np.asarray(data['PxP'])*np.pi/2
cl_gigj = np.asarray(data[name_gigj].transpose())*2*np.pi/(ell*(ell+1))

## Covariance matrix : tomographic case (ten bins) ##
cov_tot = np.zeros((len(noises), nbins, nbins, len(ell)))
for k in range(len(noises)):
    for i in range(nbins):
        for j in range(nbins):
            if j!=i:
                cl_gigj = np.asarray(data['W'+str(i+1)+'xW'+str(j+1)][0:lmax
                          ].transpose())*((2*np.pi/(ell*(ell+1))))
                cov_tot[k, i, j ,:] = (1/((2*ell+1)*fsky))*((cl_x[i]*cl_x[j])
                +(cl_gigj)*(cl_p + noises[k]))
                
                if (np.isnan(cl_x.any()) == True):
                    print('error: Nan')
            else: 
                cov_tot[k,i, j, :] = (1/((2*ell+1)*fsky))*((cl_x[j]**2)
                +(cl_g[j] + n_gg)*(cl_p + noises[k]))
            
inv_cov = np.zeros((len(noises), nbins, nbins, len(ell)))
for i in range(len(ell)):
    inv_cov[:, :, :, i] = np.linalg.inv(cov_tot[:, :, :, i])
           
## S/N: tomographic case ##
sn2 = np.zeros((len(noises), len(ell)))
for k in range(len(noises)):
    for i in range(len(ell)):
        sn2[k][i] = np.asarray(cl_x)[:, i]@(np.linalg.inv(cov_tot[k, :, : ,i])
                                        )@np.asarray(cl_x)[:, i]  
    print('S/N tomographic case:', names[k], f'= {np.sqrt(sn2[k].sum()):.1f}')