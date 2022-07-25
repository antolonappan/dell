import numpy as np
import os
import mpi
import matplotlib.pyplot as plt
import pickle as pl
import toml
import healpy as hp
import curvedsky as cs
import cmb
from utils import camb_clfile,cli

from simulation import SimExperimentFG

class Filtering:
    def __init__(self,sim_lib,maskpath,beam):

        self.sim_lib = sim_lib
        self.mask = hp.ud_grade(hp.read_map(maskpath),self.sim_lib.dnside)
        self.fsky = np.average(self.mask)
        self.beam = hp.gauss_beam(np.radians(beam/60),lmax=self.sim_lib.lmax)
        self.sim_lib = sim_lib

        #importing from sim lib
        self.Tcmb = self.sim_lib.Tcmb
        self.lmax = self.sim_lib.lmax
        self.nside = self.sim_lib.dnside
        self.cl_len = self.sim_lib.cl_len
        self.nsim = self.sim_lib.nsim

        #needed for filtering
        self.Bl = np.reshape(self.beam,(1,self.lmax+1))
        self.ninv = np.reshape(np.array((self.mask,self.mask)),(2,1,hp.nside2npix(self.nside)))

        self.lib_dir = os.path.join(self.sim_lib.outfolder,'Filtered')
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)

    @classmethod
    def from_ini(cls,ini_file):
        sim_lib = SimExperimentFG.from_ini(ini_file)
        config = toml.load(ini_file)
        fc = config['Filtering']
        maskpath = fc['maskpath']
        beam = fc['beam']
        return cls(sim_lib,maskpath,beam)

    def convolved_TEB(self,idx):
        T,E,B = self.sim_lib.get_cleaned_cmb(idx)
        hp.almxfl(T,self.beam,inplace=True)
        hp.almxfl(E,self.beam,inplace=True)
        hp.almxfl(B,self.beam,inplace=True)
        return T,E,B

    def TQU_to_filter(self,idx):
        T,E,B = self.convolved_TEB(idx)
        return hp.alm2map([T,E,B],nside=self.nside)

    @property
    def NL(self):
        nt,ne,nb = self.sim_lib.noise_spectra(500)
        return np.reshape(np.array((cli(ne[:self.lmax+1]/self.Tcmb**2),
                          cli(nb[:self.lmax+1]/self.Tcmb**2))),(2,1,self.lmax+1))



    def cinv_EB(self,idx,test=False):
        fsky = f"{self.fsky:.2f}".replace('.','p')
        fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
        if not os.path.isfile(fname):
            TQU = self.TQU_to_filter(idx)
            QU = np.reshape(np.array((TQU[1]*self.mask,TQU[2]*self.mask)),
                            (2,1,hp.nside2npix(self.nside)))/self.Tcmb
            
            iterations = [1000]
            stat_file = '' 
            if test:
                print(f"Cinv filtering is testing {idx}")
                iterations = [10]
                stat_file = os.path.join(self.lib_dir,'test_stat.txt')

            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:],
                                        self.Bl, self.ninv,QU,chn=1,itns=iterations,
                                        eps=[1e-5],ro=10,inl=self.NL,stat=stat_file)
            if not test:
                pl.dump((E,B),open(fname,'wb'))
        else:
            E,B = pl.load(open(fname,'rb'))
        
        return E,B

    def plot_cinv(self,idx):
        E,B = self.cinv_EB(idx)
        clb = cs.utils.alm2cl(self.lmax,B)
        plt.figure(figsize=(8,8))
        plt.loglog(clb,label='B')
        plt.loglog(1/self.cl_len[2,:])
 

    def wiener_EB(self,idx):
        pass

    def run_job(self):
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            eb = self.cinv_EB(i)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-cinv', dest='cinv', action='store_true', help='cinv filtering')

    args = parser.parse_args()
    ini = args.inifile[0]

    filt = Filtering.from_ini(ini)

    if args.cinv:
        filt.run_job()

    mpi.barrier()



