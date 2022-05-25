import healpy as hp
import numpy as np
import os
from utils import cli


class ExpMaps:
    
    def __init__(self,lib_dir,beam=30,lmax=1024,prefix='exp_sims_'):
        self.lib_dir = lib_dir
        self.prefix = prefix
        fwhm = np.radians(beam/60)
        self.beam = hp.gauss_beam(fwhm,pol=True,lmax=lmax).T
        self.fac  = 2.726e6 
        
    def get_sim_tlm(self,idx):
        #return hp.almxfl(hp.read_alm(os.path.join(self.lib_dir,f"{self.prefix}{idx:04d}.fits"),1),cli(self.beam[0]))/self.fac
        return hp.read_alm(os.path.join(self.lib_dir,f"{self.prefix}{idx:04d}.fits"),1)/self.fac
    
    def get_sim_elm(self,idx):
        #return hp.almxfl(hp.read_alm(os.path.join(self.lib_dir,f"{self.prefix}{idx:04d}.fits"),2),cli(self.beam[1]))/self.fac
        return hp.read_alm(os.path.join(self.lib_dir,f"{self.prefix}{idx:04d}.fits"),2)/self.fac
    
    def get_sim_blm(self,idx):
        #return hp.almxfl(hp.read_alm(os.path.join(self.lib_dir,f"{self.prefix}{idx:04d}.fits"),3),cli(self.beam[2]))/self.fac
        return hp.read_alm(os.path.join(self.lib_dir,f"{self.prefix}{idx:04d}.fits"),3)/self.fac