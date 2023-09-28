import numpy as np
import os
import mpi
import matplotlib.pyplot as plt
import pickle as pl
import toml
import healpy as hp
import curvedsky as cs
from utils import cli
from utils import ini_full
from tqdm import tqdm

from simulation import SimExperimentFG

class Filtering:
    """
    Filtering class for component separated CMB Maps

    Parameters
    ----------
    sim_lib : object : simulation.SimExperimentFG- simulaiton library
    maskpath : string : path to mask
    fullsky : bool : if True, use fullsky 
    beam : float : beam size in arcmin
    verbose : bool : if True, print verbose output
    """
    def __init__(self,sim_lib,maskpath,fullsky,beam,verbose=False):

        self.sim_lib = sim_lib
        self.mask = hp.ud_grade(hp.read_map(maskpath),self.sim_lib.dnside)
        self.fsky = np.average(self.mask)
        self.fname = ''
        self.fullsky = fullsky
        if self.fullsky:
            self.mask = np.ones(hp.nside2npix(self.sim_lib.dnside))
            self.fname = '_fullsky'
            self.fsky = 1.0

        #importing from sim lib
        self.Tcmb = self.sim_lib.Tcmb
        self.lmax = self.sim_lib.lmax
        self.nside = self.sim_lib.dnside
        self.cl_len = self.sim_lib.cl_len
        self.nsim = self.sim_lib.nsim

        #needed for filtering
        self.beam = hp.gauss_beam(np.radians(beam/60),lmax = self.lmax)
        self.Bl = np.reshape(self.beam,(1,self.lmax+1))
        self.ninv = np.reshape(np.array((self.mask,self.mask)),(2,1,hp.nside2npix(self.nside)))

        self.lib_dir = os.path.join(self.sim_lib.outfolder,f"Filtered{self.fname}")
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)

        self.verbose = verbose
        self.vprint(f"FILTERING INFO: Outfolder - {self.lib_dir}")
        self.vprint(f"FILTERING INFO: Mask path - {maskpath}")
        self.vprint(f"FILTERING INFO: fsky - {self.fsky}")
        self.vprint(f"FILTERING INFO: Beam - {beam} arcmin")
        print(f"FILTERING object with {'out' if self.sim_lib.noFG else ''} FG: Loaded")
    
    def vprint(self,txt):
        """
        print the text if verbose is True

        Parameters
        ----------
        txt : string : text to print
        """

        if self.verbose:
            print(txt)

    @classmethod
    def from_ini(cls,ini_file,verbose=False):
        """
        class method to create Filtering object from ini file

        Parameters
        ----------
        ini_file : string : path to ini file
        verbose : bool : if True, print verbose output
        """
        sim_lib = SimExperimentFG.from_ini(ini_file)
        config = toml.load(ini_full(ini_file))
        fc = config['Filtering']
        maskpath = fc['maskpath']
        fullsky = bool(fc['fullsky'])
        beam = float(fc['beam'])
        return cls(sim_lib,maskpath,fullsky,beam,verbose)

    def convolved_TEB(self,idx):
        """
        convolve the component separated map with the beam

        Parameters
        ----------
        idx : int : index of the simulation
        """
        T,E,B = self.sim_lib.get_cleaned_cmb(idx)
        hp.almxfl(T,self.beam,inplace=True)
        hp.almxfl(E,self.beam,inplace=True)
        hp.almxfl(B,self.beam,inplace=True)
        return T,E,B

    def TQU_to_filter(self,idx):
        """
        Change the convolved ALMs to MAPS

        Parameters
        ----------
        idx : int : index of the simulation
        """
        T,E,B = self.convolved_TEB(idx)
        return hp.alm2map([T,E,B],nside=self.nside)

    @property
    def NL(self):
        """
        array manipulation of noise spectra obtained by ILC weight
        for the filtering process
        """
        nt,ne,nb = self.sim_lib.noise_spectra(self.sim_lib.nsim)
        return np.reshape(np.array((cli(ne[:self.lmax+1]*self.beam**2),
                          cli(nb[:self.lmax+1]*self.beam**2))),(2,1,self.lmax+1))

    def cinv_EB(self,idx,test=False):
        """
        C inv Filter for the component separated maps

        Parameters
        ----------
        idx : int : index of the simulation
        test : bool : if True, run the filter for 10 iterations
        """
        fsky = f"{self.fsky:.2f}".replace('.','p')
        fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
        if not os.path.isfile(fname):
            TQU = self.TQU_to_filter(idx)
            QU = np.reshape(np.array((TQU[1]*self.mask,TQU[2]*self.mask)),
                            (2,1,hp.nside2npix(self.nside)))/self.Tcmb
            
            iterations = [1000]
            stat_file = '' 
            if test:
                self.vprint(f"Cinv filtering is testing {idx}")
                iterations = [10]
                stat_file = os.path.join(self.lib_dir,'test_stat.txt')

            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:],
                                        self.Bl, self.ninv,QU,chn=1,itns=iterations,filter="",
                                        eps=[1e-5],ro=10,inl=self.NL,stat=stat_file)
            if not test:
                pl.dump((E,B),open(fname,'wb'))
        else:
            E,B = pl.load(open(fname,'rb'))
        
        return E,B

    def plot_cinv(self,idx):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        _,B = self.cinv_EB(idx)
        _,_,nb = self.sim_lib.noise_spectra(self.sim_lib.nsim)
        clb = cs.utils.alm2cl(self.lmax,B)
        plt.figure(figsize=(8,8))
        plt.loglog(clb,label='B')
        plt.loglog(1/(self.cl_len[2,:]  + nb))

    def wiener_EB(self,idx):
        """
        Not implemented yet
        useful for delensing
        """
        E, B = self.cinv_EB(idx)
        pass

    def run_job_mpi(self):
        """
        MPI job for filtering
        """
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            eb = self.cinv_EB(i)

    def run_job(self):
        """
        MPI job for filtering
        """
        jobs = np.arange(self.sim_lib.nsim)
        for i in tqdm(jobs, desc='Cinv filtering', unit='sim'):
            eb = self.cinv_EB(i)
            del eb

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
