import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from itertools import combinations
import pandas as pd
from quest import Reconstruction


class SNR:
    def __init__(self,rec_ini,nbins=10,lmax=1000,fsky=0.36,fsky_iswphi=0.80):
        self.nbins = nbins
        self.lmax = lmax
        self.fsky = fsky
        self.fsky_iswphi = fsky_iswphi
        rec = Reconstruction.from_ini(rec_ini)
        self.ell = np.arange(2,self.lmax+1)
        self.Lfac =  (self.ell*(self.ell+1))/(2*np.pi)
        MCN0 = (rec.MCN0()/rec.response_mean()**2)
        self.MCN0 = MCN0[2:self.lmax+1] 

        self.n_gg = (2.8e-9)*nbins

    def snr_tot_survey(self):
        filename_tot = '../Data/lite_euclid_camb_tot_nl5_lmax1000_ScalarCovCls.txt'
        cl_x_tot = np.loadtxt(filename_tot, usecols = [12]
                            )/np.sqrt(self.ell*(self.ell+1))/self.Lfac    
        cl_p_tot = np.loadtxt(filename_tot, dtype = float, unpack = True, 
                            usecols = [11])/(self.ell*(self.ell+1))/self.Lfac
        cl_g_tot = np.loadtxt(filename_tot, dtype = float,
                    unpack = True, usecols = [16])/self.Lfac
        
        sn_tot_survey = (self.fsky*(2*self.ell+1)*(cl_x_tot)**2)/((cl_x_tot)**2 + (cl_g_tot + 
                        self.n_gg/self.nbins)*(cl_p_tot+self.MCN0))
        
        return sn_tot_survey
    
    def snr_total(self):
        return np.sqrt(self.snr_tot_survey().sum())
    
    ## snr ISWx lensing convergence #
    def sn2r_iswphi(self):
        filename_tot = '../Data/lite_euclid_camb_tot_nl5_lmax1000_ScalarCovCls.txt'   
        cl_p_tot = np.loadtxt(filename_tot, dtype = float, unpack = True, 
                            usecols = [11])/(self.ell*(self.ell+1))/self.Lfac
        cl_tt = np.loadtxt(filename_tot, dtype = float,
                    unpack = True, usecols = [1])/self.Lfac
        cl_tp = np.loadtxt(filename_tot, dtype = float,
                    unpack = True, usecols = [3])/np.sqrt(self.ell*(self.ell+1))/self.Lfac 
        sn2_iswphi = (self.fsky_iswphi*(2*self.ell+1)*(cl_tp)**2)/((cl_tp)**2 + cl_tt*(cl_p_tot + self.MCN0))
        
        return sn2_iswphi
    
    def snr_iswphi(self):
        return np.sqrt(self.sn2r_iswphi().sum())
    
    
    def snr_tomo_survey(self):
        filename = '../Data/lite_euclid_camb_bins_step_nl5_lmax1000_ScalarCovCls.txt'
        name_x = ['PxW{}'.format(i) for i in range(1,self.nbins+1)]
        name_a = ['W{}'.format(i)+'xW{}'.format(i) for i in range(1,self.nbins+1)]
        name_gigj = []
        index = np.asarray(sorted((set(combinations(np.arange(1, self.nbins+1,1), r =2)))), 
                                dtype = int)       
        for i in range(len(index)):
            name_gigj.append('W'+str(index[i][0])+'xW'+str(index[i][1]))
            
        data = pd.read_csv(filename, delim_whitespace=True, index_col=0)
        cl_x = (np.asarray(data[name_x].transpose()))/np.sqrt(self.ell*(self.ell+1))/self.Lfac
        cl_g = np.asarray(data[name_a].transpose())/self.Lfac
        cl_p = np.asarray(data['PxP'])/(self.ell*(self.ell+1))/self.Lfac
        cl_gigj = np.asarray(data[name_gigj].transpose())/self.Lfac

        ## Covariance matrix : tomographic case (ten bins) ##
        cov_tot = np.zeros((self.nbins, self.nbins, len(self.ell)))
        for i in range(self.nbins):
            for j in range(self.nbins):
                if j!=i:
                    cl_gigj = np.asarray(data['W'+str(i+1)+'xW'+str(j+1)][0:self.lmax
                            ].transpose())*((2*np.pi/(self.ell*(self.ell+1))))
                    cov_tot[ i, j ,:] = (1/((2*self.ell+1)*self.fsky))*((cl_x[i]*cl_x[j])
                                                    +(cl_gigj)*(cl_p + self.MCN0))

                    if (np.isnan(cl_x.any()) == True):
                        print('error: Nan')
                else: 
                    cov_tot[i, j, :] = (1/((2*self.ell+1)*self.fsky))*((cl_x[j]**2)
                    +(cl_g[j] + self.n_gg)*(cl_p + self.MCN0))
                
        inv_cov = np.zeros((self.nbins, self.nbins, len(self.ell)))
        for i in range(len(self.ell)):
            inv_cov[ :, :, i] = np.linalg.inv(cov_tot[ :, :, i])
                
        ## S/N: tomographic case ##
        sn2 = np.zeros( len(self.ell))
        for i in range(len(self.ell)):
            sn2[i] = np.asarray(cl_x)[:, i]@(np.linalg.inv(cov_tot[ :, : ,i])
                                                )@np.asarray(cl_x)[:, i]  
        return sn2

    def snr_tomo(self):
        return np.sqrt(self.snr_tomo_survey().sum())
