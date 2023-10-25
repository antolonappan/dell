import os
import healpy as hp
import mpi
import numpy as np
import matplotlib.pyplot as plt
import curvedsky as cs
from utils import camb_clfile,hash_array,ini_full
import pickle as pl
from filtering import Filtering
import toml
from tqdm import tqdm
import analysis as ana
import binning
import pandas as pd
import seaborn as sns

class Reconstruction:
    """
    Class to reconstruct the lensing potentials from the filtered CMB fields.

    filt_lib: class object : filtering.Filtering 
    Lmax: int : maximum multipole of the reconstruction
    rlmin: int : minimum multipole of CMB for the reconstruction
    rlmax: int : maximum multipole of CMB for the reconstruction
    cl_unl: str : path to the unlensed CMB power spectrum
    nbins: int : number of bins for the multipole binning
    tp_nbins: int : number of bins for the multipole binning for the ISW
    verbose: bool : print the information of the reconstruction
    """

    def __init__(self,filt_lib,Lmax,rlmin,rlmax,cl_unl,nbins,tp_nbins,bin_opt='',verbose=False):
        self.filt_lib = filt_lib
        self.Lmax = Lmax
        self.rlmin = rlmin
        self.rlmax = rlmax
        self.lib_dir = os.path.join(self.filt_lib.sim_lib.outfolder,
                                    f'Reconstruction_{self.rlmin}_{self.rlmax}{self.filt_lib.fname}')
        self.in_dir = os.path.join(self.lib_dir,'input')
        self.plm_dir = os.path.join(self.lib_dir,'plm')
        self.n0_dir = os.path.join(self.lib_dir,'N0')
        self.rdn0_dir = os.path.join(self.lib_dir,'RDN0')
        self.rp_dir = os.path.join(self.lib_dir,'response')
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
            os.makedirs(self.in_dir,exist_ok=True)
            os.makedirs(self.plm_dir,exist_ok=True)
            os.makedirs(self.n0_dir,exist_ok=True)
            os.makedirs(self.rdn0_dir,exist_ok=True)
            os.makedirs(self.rp_dir,exist_ok=True)
        
        self.mask = self.filt_lib.mask
        self.fsky = self.filt_lib.fsky
        self.nside = self.filt_lib.nside
        self.cl_len = self.filt_lib.cl_len[:,:self.rlmax+1]
        self.cl_pp = camb_clfile(cl_unl)['pp'][:self.Lmax+1]
        self.cl_unl = camb_clfile(cl_unl)
        self.Tcmb = self.filt_lib.Tcmb
        self.nsim = self.filt_lib.nsim
        self.bin_opt = bin_opt

        self.norm = self.get_norm

        self.L = np.arange(self.Lmax+1)
        self.Lfac = (self.L*(self.L+1.))**2/(2*np.pi)
        self.mf_array = np.arange(400,500)

        self.nbins = nbins
        self.tp_nbins = tp_nbins
        self.binner = binning.multipole_binning(self.nbins,lmin=2,lmax=self.Lmax,spc=self.bin_opt)
        self.binnertp = binning.multipole_binning(self.tp_nbins,lmin=2,lmax=self.Lmax)
        self.B = self.binner.bc
        self.Btp = self.binnertp.bc
        self.Bfac = (self.B*(self.B+1.))**2/(2*np.pi)
        if self.fsky>0.8:
            N1_extra = f'_{self.fsky:.2f}'
        else:
            N1_extra = ''
        N1_file = os.path.join(self.lib_dir,f'n1{N1_extra}.pkl')
        self.N1 = pl.load(open(N1_file,'rb')) if os.path.isfile(N1_file) else np.zeros(self.Lmax+1)
        self.verbose = verbose

        self.vprint(f"QUEST INFO: Maximum L - {self.Lmax}")
        self.vprint(f"QUEST INFO: Minimum CMB multipole - {self.rlmin}")
        self.vprint(f"QUEST INFO: Maximum CMB multipole - {self.rlmax}")
        self.vprint(f"QUEST INFO: N1 file found - {not np.all(self.N1 == 0)}")
        print(f"QUEST object with {'out' if self.filt_lib.sim_lib.noFG else ''} FG: Loaded")

    @classmethod
    def from_ini(cls,ini_file,verbose=False):
        """
        Load the reconstruction object from a ini file.
        """
        filt_lib = Filtering.from_ini(ini_file)
        config = toml.load(ini_full(ini_file))
        rc = config['Reconstruction']
        Lmax = rc['phi_lmax']
        rlmin = rc['cmb_lmin']
        rlmax = rc['cmb_lmax']
        cl_unl = rc['cl_unl']
        nbins = rc['nbins']
        tp_nbins = rc['nbins_tp']
        bin_opt = rc['bin_opt']
        return cls(filt_lib,Lmax,rlmin,rlmax,cl_unl,nbins,tp_nbins,bin_opt,verbose)
    
    def vprint(self,txt):
        """
        print only if verbose is true

        txt: str : text to print
        """
        if self.verbose:
            print(txt)
        
    def bin_cell(self,arr):
        """
        binning function for the multipole bins

        arr: array : array to bin
        """
        return binning.binning(arr,self.binner)
    
    def bin_cell_tp(self,arr):
        """
        binning function for the multipole bins

        arr: array : array to bin
        """
        return binning.binning(arr,self.binnertp)
    
    @property
    def __kfac__(self):
        """
        Calculate the factor of wiener filter
        """
        nhl = self.MCN0()/self.response_mean()**2
        fl = self.cl_pp/(self.cl_pp+ nhl )
        fl[0] = 0
        fl[1] = 0
        return fl

    @property
    def __observed_spectra__(self):
        """
        Calculate the expected observed spectra using ILC noise
        and effective beam
        """

        ocl = self.cl_len.copy()
        nt,ne,nb = self.filt_lib.sim_lib.noise_spectra(self.filt_lib.sim_lib.nsim)

        ocl[0,:] += nt[:self.rlmax+1]
        ocl[1,:] += ne[:self.rlmax+1]
        ocl[2,:] += nb[:self.rlmax+1]

        return ocl

    def test_obs_for_norm(self):
        """
        Test the observed spectra for the normalization is 
        visually acceptable.
        """
        obs = self.__observed_spectra__.copy()
        cmb,_,_ = self.filt_lib.sim_lib.get_cmb_alms(0)
        plt.figure(figsize=(8,8))
        plt.loglog(hp.alm2cl(cmb[1])/self.Tcmb**2,label='HILC EE')
        plt.loglog(hp.alm2cl(cmb[2])/self.Tcmb**2,label='HILC BB')
        plt.loglog(self.cl_len[2,:],label='BB')
        plt.loglog(obs[1,:],label='EE + FG res + ILC noise/ILC beam^2')
        plt.loglog(obs[2,:], label='BB + FG res + ILC noise/ILC beam^2')
        plt.axhline(np.radians(2.16/60)**2 /self.Tcmb**2)
        plt.xlim(100,None)
        plt.legend(fontsize=12)


    @property
    def get_norm(self):
        """
        Normalization of the reconstruction.
        """
        ocl = self.__observed_spectra__
        Ag, Ac = cs.norm_quad.qeb('lens',self.Lmax,self.rlmin,
                                  self.rlmax,self.cl_len[1,:],
                                  ocl[1,:],ocl[2,:])  
        del Ac
        return Ag

    def get_phi(self,idx):
        """
        Reconstruct the potential using filtered Fields.

        idx: int : index of the Reconstruction
        """
        fname = os.path.join(self.plm_dir,f"phi_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E,B = self.filt_lib.cinv_EB(idx)
            glm, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                       self.cl_len[1,:self.rlmax+1],
                                       E[:self.rlmax+1,:self.rlmax+1],
                                       B[:self.rlmax+1,:self.rlmax+1])
            del(clm)
            glm *= self.norm[:,None]
            pl.dump(glm,open(fname,'wb'))
            return glm


    def N0_sim(self,idx):
        """
        Calculate the N0 bias from the Reconstructed potential using filtered Fields
        with different CMB fields. If E modes is from ith simulation then B modes is 
        from (i+1)th simulation

        idx: int : index of the N0
        """
        myidx = np.pad(np.arange(self.nsim),(0,1),'constant',constant_values=(0,0))
        fname = os.path.join(self.n0_dir,f"N0_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E1,B1 = self.filt_lib.cinv_EB(myidx[idx])
            E2,B2 = self.filt_lib.cinv_EB(myidx[idx+1])
            glm1, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                       self.cl_len[1,:self.rlmax+1],
                                       E1[:self.rlmax+1,:self.rlmax+1],
                                       B2[:self.rlmax+1,:self.rlmax+1])
            glm2, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                        self.cl_len[1,:self.rlmax+1],
                                        E2[:self.rlmax+1,:self.rlmax+1],
                                        B1[:self.rlmax+1,:self.rlmax+1])
            glm1 *= self.norm[:,None]
            glm2 *= self.norm[:,None]
            
            glm = glm1 + glm2
            n0cl = cs.utils.alm2cl(self.Lmax,glm)/(2*self.fsky) # type: ignore 
            pl.dump(n0cl,open(fname,'wb'))
            return n0cl

    def MCN0(self,n=500):
        """
        Calculate the Monte Carlo N0 bias

        n: int : number of simulations
        """
        if n > self.nsim:
            n = self.nsim

        def get_N0_mean(n):
            m = np.zeros(self.Lmax+1,dtype=np.float64)
            for i in tqdm(range(n), desc='cross spectra stat',unit='simulation'):
                m += self.N0_sim(i)
            return m/n
        
        fname = os.path.join(self.lib_dir,f"MCN0_{n}_fsky_{self.fsky:.2f}.pkl")
        if os.path.isfile(fname):
            arr = pl.load(open(fname,'rb'))
        else:
            arr = get_N0_mean(n)
            pl.dump(arr,open(fname,'wb'))
        return arr

    def RDN0(self,idx):
        """
        Calculate the Realization Dependent N0 bias

        idx: int : index of the RDN0
        """

        fname = os.path.join(self.rdn0_dir,f"RDN0_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            myidx = np.append(np.arange(self.nsim),np.arange(2))
            sel = np.where(myidx == idx)[0]
            myidx = np.delete(myidx,sel)

            E0,B0 = self.filt_lib.cinv_EB(idx)

            mean_rdn0 = []

            for i in tqdm(range(100),desc=f'RDN0 for simulation {idx}', leave=True, unit='sim',position=1):
                E1,B1 = self.filt_lib.cinv_EB(myidx[i])
                E2,B2 = self.filt_lib.cinv_EB(myidx[i+1])
                # E_0,B_1
                glm1, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                            self.cl_len[1,:self.rlmax+1],
                                            E0[:self.rlmax+1,:self.rlmax+1],
                                            B1[:self.rlmax+1,:self.rlmax+1])
                del clm
                # E_1,B_0
                glm2, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                            self.cl_len[1,:self.rlmax+1],
                                            E1[:self.rlmax+1,:self.rlmax+1],
                                            B0[:self.rlmax+1,:self.rlmax+1])
                del clm
                # E_1,B_2
                glm3, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                            self.cl_len[1,:self.rlmax+1],
                                            E1[:self.rlmax+1,:self.rlmax+1],
                                            B2[:self.rlmax+1,:self.rlmax+1])
                del clm
                # E_2,B_1
                glm4, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,    
                                            self.cl_len[1,:self.rlmax+1],
                                            E2[:self.rlmax+1,:self.rlmax+1],
                                            B1[:self.rlmax+1,:self.rlmax+1])
                del (clm,E1,B1,E2,B2)


                glm1 *= self.norm[:,None]
                glm2 *= self.norm[:,None]
                glm3 *= self.norm[:,None]
                glm4 *= self.norm[:,None]

                first_four = cs.utils.alm2cl(self.Lmax, glm1 + glm2)/(self.fsky) #type: ignore
                del (glm1,glm2)
                second_last = cs.utils.alm2cl(self.Lmax, glm3)/(self.fsky) #type: ignore
                last = cs.utils.alm2cl(self.Lmax, glm3,glm4)/(self.fsky) #type: ignore
                del (glm3,glm4)

                mean_rdn0.append(first_four - second_last - last)
                del (first_four,second_last,last)

            del (E0,B0)
            rdn0 = np.mean(mean_rdn0,axis=0)
            pl.dump(rdn0,open(fname,'wb'))
            return rdn0
    
    def RDN0_mean(self,n=400):
        return np.mean([self.RDN0(i) for i in range(n)], axis=0)

    def mean_field(self):
        """
        Calcualte the Mean Field bias.
        """
        fname = os.path.join(self.lib_dir,f"MF_fsky_{self.fsky:.2f}_{hash_array(self.mf_array)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            arr = np.zeros((self.Lmax+1,self.Lmax+1),dtype=complex)
            for i in tqdm(self.mf_array,desc="Calculating Mean Field",unit='Simulation'):
                arr += self.get_phi(i)
            arr /= len(self.mf_array)
            if mpi.rank == 0:
                pl.dump(arr,open(fname,'wb'))
            return arr

    def mean_field_cl(self):
        """
        Mean Field bias cl
        """
        n = len(self.mf_array)
        arr =  cs.utils.alm2cl(self.Lmax,self.mean_field())/self.fsky
        arr  += (1/n) * (arr+self.cl_pp[:self.Lmax+1])
        return arr

    def wf_phi(self,idx,kappa=False,filt_lmax=None):
        """
        Calculate the wiener filtered phi.

        idx: int : index of the simulation
        """
        phi = self.get_phi(idx) - self.mean_field()
        phi =  cs.utils.almxfl(self.Lmax,self.Lmax,phi,self.__kfac__)
        if kappa:
            fl = self.L * (self.L + 1)/2
            if filt_lmax is not None:
                fl[filt_lmax:] = 0
            klm = cs.utils.almxfl(self.Lmax,self.Lmax,phi,fl)
            return cs.utils.hp_alm2map(self.nside,self.Lmax,self.Lmax,klm)
        else:
            return phi
    
    def deflection_angle(self,idx):
        """
        Calculate the deflection field in spherical harmonics.
        \sqrt(L(L+1)) \phi

        idx: int : index of the simulation
        """
        wfphi = self.wf_phi(idx)
        dl = np.sqrt(np.arange(self.Lmax + 1, dtype=float) * np.arange(1, self.Lmax + 2))
        dl[:10] = 0
        return cs.utils.almxfl(self.Lmax,self.Lmax,wfphi,dl)

    def deflection_map(self,idx):
        """
        Calculate the deflection map.

        idx: int : index of the simulation
        """
        alm = self.deflection_angle(idx)
        return cs.utils.hp_alm2map(self.nside,self.Lmax,self.Lmax,alm)
    
    def get_phi_cl(self,idx):
        """
        Get the cl of the reconstructed potential.

        idx: int : index of the simulation
        """
        if idx in self.mf_array:
            raise ValueError("Simulation already in mean field array")
        else:
            return cs.utils.alm2cl(self.Lmax,self.get_phi(idx)-self.mean_field())/self.fsky


    def get_input_phi_sim(self,idx,kappa=False,filt_lmax=None):
        """
        Get the masked input potential alms. If the total no of sim
        is less than 200 then the input phi is constant. Otherwise
        it is varying.

        idx: int : index of the simulation
        """
        if self.fsky > 0.8:
            extra = f"_{self.fsky:.2f}"
        else:
            extra = ''
        if self.nsim <200:
            self.vprint("input phi is constant")
            dir_ = "/global/cfs/cdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps_c/MASS"
            fname = os.path.join(dir_,f"phi_sims{extra}.fits")
            fnamet = os.path.join(self.in_dir,f"phi_sims_c.pkl")
        else:
            self.vprint("input phi is from variying")
            dir_ = "/global/cfs/cdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps/MASS"
            fname = os.path.join(dir_,f"phi_sims_{idx:04d}.fits")
            fnamet = os.path.join(self.in_dir,f"phi_sims{extra}_{idx:04d}.pkl")
        if os.path.isfile(fnamet) and (not kappa):
            return pl.load(open(fnamet,'rb'))
        else:
            plm = hp.read_alm(fname)
            fl = self.L * (self.L + 1)/2
            if filt_lmax is not None:
                fl[filt_lmax:] = 0
            klm = hp.almxfl(plm,fl)
            kmap = hp.alm2map(klm,nside=self.nside)*self.mask
            if kappa:
                return kmap
            klm_n = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,kmap)
            plm_n = cs.utils.almxfl(self.Lmax,self.Lmax,klm_n,1/fl)
            pl.dump(plm_n,open(fnamet,'wb')) 
            return plm_n

    def get_input_phi_cl(self,idx):
        """
        Get the cl of the input potential.

        idx: int : index of the simulation
        """
        return cs.utils.alm2cl(self.Lmax,self.get_input_phi_sim(idx))


    def get_cl_phi_inXout(self,idx):
        """
        get input X output potential

        idx: int : index of the simulation
        """

        almi = self.get_input_phi_sim(idx)
        almo = self.get_phi(idx) #- self.mean_field()
        return cs.utils.alm2cl(self.Lmax,almi,almo)/self.fsky
    
    def response(self,idx):
        """
        Calculate the response
        r = cl^{cross} / cl^{input}

        idx: int : index of the simulation
        """

        fname = os.path.join(self.rp_dir,f"response_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            almi = self.get_input_phi_sim(idx)
            almo = self.get_phi(idx) # - self.mean_field()
            r =  cs.utils.alm2cl(self.Lmax,almi,almo)/cs.utils.alm2cl(self.Lmax,almi)
            r[0] = 0
            r[1] = 0
            pl.dump(r,open(fname,'wb'))
            return r
    
    def response_mean(self):
        """
        Mean of response for all simulations
        """
        fname = os.path.join(self.lib_dir,f"response_fsky_{self.fsky:.2f}_mean.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            r = np.zeros(self.Lmax+1)
            for i in tqdm(range(self.nsim),desc="Calculating Response",unit='Simulation'):
                r += self.response(i)
            r /= self.nsim
            pl.dump(r,open(fname,'wb'))
            return r

    
    def get_qcl(self,idx,n1=True,rdn0=False):
        """
        Get the cl_phi = cl_recon - mean field - N0 - N1

        idx: int : index of the simulation
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        cl = self.get_phi_cl(idx)
        if n1 :
            cl -=  self.N1 
        if rdn0:
            cl  -= self.RDN0(idx)
        else:
            cl -= self.MCN0()
        return cl
  
    def get_qcl_wR(self,idx,n1=True,rdn0=False):
        """
        Get the cl_phi = (cl_recon -  mean field - N0 - N1)/ response**2

        idx: int : index of the simulation
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        if rdn0:
            return self.get_qcl(idx,n1,rdn0)/self.response_mean()**2   - ((self.RDN0(idx)/self.response_mean()**2)+self.cl_pp)/100
        else:
            return self.get_qcl(idx,n1,rdn0)/self.response_mean()**2   - ((self.MCN0()/self.response_mean()**2)+self.cl_pp)/100

    def get_qcl_wR_stat(self,n=400,ret='dl',n1=True,rdn0=True,do_bining=True):
        """
        Get the total cl_phi

        n: int : number of simulations
        ret: str : 'dl' or 'cl'
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        bb = '' if do_bining else 'nb'

        fname = os.path.join(self.lib_dir,f"qclSTAT_fsky_{self.fsky:.2f}_nbin_{self.nbins}{self.bin_opt}{bb}_n_{n}_ret_{ret}_n1_{n1}_rd_{rdn0}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            if ret == 'cl':
                lfac = 1.0
            elif ret == 'dl':
                lfac = self.Lfac
            else:
                raise ValueError
            cl = []
            for i in tqdm(range(n), desc='qcl stat',unit='simulation'):
                if do_bining:
                    cl.append(self.bin_cell(self.get_qcl_wR(i,n1,rdn0)*self.Lfac))
                else:
                    cl.append(self.get_qcl_wR(i,n1,rdn0)*self.Lfac)
            
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl
            
    def bin_corr(self,n=400,ret='cl',n1=True,rdn0=False):
        """
        Get the correlation matrix of the total cl_phi

        n: int : number of simulations
        ret: str : 'dl' or 'cl'
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        s = self.get_qcl_wR_stat(n=n,ret=ret,n1=n1,rdn0=rdn0)
        df = pd.DataFrame(s)
        df.columns = self.B.astype(np.int)
        corr = df.corr()
        return corr
    
    def get_tXphi(self,idx):
        """
        Get the Cl_{temp, phi}

        idx: int : index of the simulation
        """
        clpp = self.cl_unl['pp'][:self.Lmax+1]
        cltt = self.cl_unl['tt'][:self.Lmax+1]
        cltp = self.cl_unl['tp'][:self.Lmax+1]
        Plm = self.get_input_phi_sim(idx)
        Tlm = cs.utils.gauss2alm_const(self.Lmax,clpp,cltt,cltp,Plm)
        del Plm
        tmap = cs.utils.hp_alm2map(self.nside,self.Lmax,self.Lmax,Tlm[1])*self.mask
        del Tlm
        Tlm = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,tmap)
        Plm = self.get_phi(idx) - self.mean_field()
        return cs.utils.alm2cl(self.Lmax,Tlm,Plm)/self.fsky
        
    def tXphi_stat(self,n,ret='cl'):
        """
        Get the total Cl_{temp, phi}

        n: int : number of simulations
        ret: str : 'dl' or 'cl'
        """

        if ret == 'cl':
            lfac = np.ones(self.Lmax+1)
            fname = os.path.join(self.lib_dir,f"tXphi_{n}_{hash_array(self.Btp)}.pkl")
        elif ret == 'dl':
            lfac = (self.L*(self.L+1))**2 / 2/(2*np.pi)
            fname = os.path.join(self.lib_dir,f"tXphi_dl_{n}_{hash_array(self.Btp)}.pkl")

        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            cl = []
            for i in tqdm(range(n), desc='Calculating TempXphi',unit='simulation'):
                cl.append(self.bin_cell_tp(self.get_tXphi(i)*lfac))
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl

    def plot_bin_cor(self,n=400,ret='cl',n1=True,rdn0=False):
        """
        Plot the correlation matrix of the total cl_phi

        n: int : number of simulations
        ret: str : 'dl' or 'cl'
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        corr = self.bin_corr(n=n,ret=ret,n1=n1,rdn0=rdn0)
        plt.figure(figsize=(10,10))
        ax = sns.heatmap(corr)

    def plot_qcl_stat(self,n=400,n1=True,rdn0=False):
        """
        Plot the mean and std of cl_phi

        n: int : number of simulations
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        stat = self.get_qcl_wR_stat(n=n,n1=n1,rdn0=rdn0)
        plt.figure(figsize=(8,7))
        plt.loglog(self.cl_pp*self.Lfac,label='Fiducial',c='grey',lw=2)
        plt.loglog(self.Lfac*(self.MCN0()/self.response_mean()**2 ),label='MCN0',c='r')
        plt.loglog(self.Lfac*self.N1,label='MCN1',c='g')
        plt.loglog(self.Lfac*self.mean_field_cl(),label='Mean Field',c='b')
        plt.errorbar(self.B,stat.mean(axis=0),yerr=stat.std(axis=0),fmt='o',c='k',ms=6,capsize=2,label='Reconstructed')
        plt.xlim(2,600)
        plt.legend(ncol=2, fontsize=20)
        plt.xlabel('L',fontsize=20)
        plt.ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.savefig(f"clpp.pdf",bbox_inches='tight',dpi=300)
    

    def plot_tXphi_stat(self,n):
        """
        Plot the mean and std of cl_{temp, phi}

        n: int : number of simulations
        """
        lfac = (self.L*(self.L+1))**2 / 2/(2*np.pi)
        cl = self.tXphi_stat(n,ret='dl')
        plt.figure(figsize=(8,6))
        plt.plot(self.L,self.cl_unl['tp'][:self.Lmax+1]*lfac)
        plt.errorbar(self.Btp,cl.mean(axis=0)/.9 ,yerr=cl.std(axis=0),fmt='o')
        plt.semilogy()
        plt.xlim(2,100)
        plt.ylabel('$[\ell(\ell+1)]^{2} C_{\ell}^{\Theta \phi} / 2 \pi$',fontsize=20)
        plt.xlabel('$\ell$',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    
    def SNR_phi(self,n=400,n1=True,rdn0=False):
        """
        Calculate the SNR of the reconstructed potential

        n: int : number of simulations
        n1: bool : if True subtract N1
        rdn0: bool : if True subtract RDN0 else subtract MCN0
        """
        cl_pp = self.get_qcl_wR_stat(n,'cl',n1=n1,rdn0=rdn0)
        stat = ana.statistics(ocl=1.,scl=cl_pp)
        stat.get_amp(fcl=cl_pp.mean(axis=0))
        return 1/stat.sA

    def SNR_tp(self,n):
        """
        Calculate the SNR of the ISW

        n: int : number of simulations
        """

        cltp = self.tXphi_stat(n,ret='dl')[:,:]
        stat = ana.statistics(ocl=1.,scl=cltp)
        stat.get_amp(fcl=cltp.mean(axis=0))
        return 1/stat.sA
    
    def job_phi(self):
        """
        MPI job for the potential reconstruction.
        """
        job = np.arange(self.nsim)
        if mpi.size > 1:
            for i in job[mpi.rank::mpi.size]:
                phi = self.get_phi(i)
            mpi.barrier()
        else:
            for i in tqdm(job,desc='Phi reconstruction', unit='sim'):
                phi = self.get_phi(i)
                del phi
        
    
    def job_MCN0(self):
        """
        MPI job for the potential reconstruction with different CMB fields.
        """
        job = np.arange(self.nsim)
        if mpi.size > 1:
            for i in job[mpi.rank::mpi.size]:
                phi = self.N0_sim(i)
                del phi
            mpi.barrier()
        else:
            for i in tqdm(job,desc='MCN0', unit='sim'):
                phi = self.N0_sim(i)
                del phi
        
    
    def job_RDN0(self):
        """
        MPI job for the potential reconstruction with different CMB fields.
        """
        job = np.arange(self.nsim)
        if mpi.size > 1:
            for i in job[mpi.rank::mpi.size]:
                rdn0 = self.RDN0(i)
                del rdn0
            mpi.barrier()
        else:
            for i in tqdm(job,desc='RDN0', unit='sim',position=0):
                rdn0 = self.RDN0(i)
                del rdn0

    def job_response(self):
        """
        MPI job for the response calculation.
        """
        job = np.arange(self.nsim)
        if mpi.size > 1:
            for i in job[mpi.rank::mpi.size]:
                Null = self.response(i)
            mpi.barrier()
        else:
            for i in tqdm(job,desc='Response', unit='sim'):
                Null = self.response(i)
                del Null
    
    def job_input_phi(self):
        """
        MPI job for the input potential reconstruction.
        """
        job = np.arange(self.nsim)
        if mpi.size > 1:
            for i in job[mpi.rank::mpi.size]:
                phi = self.get_input_phi_sim(i)
            mpi.barrier()
        else:
            for i in tqdm(job,desc='Input Phi', unit='sim'):
                phi = self.get_input_phi_sim(i)
                del phi



class N1:
    """
    Class to calculate N1

    c_phi_ini: str : the ini file for the CMB potential reconstruction with 
                     the same input potential
    v_phi_ini: str : the ini file for the CMB potential reconstruction with
                     the variying input potential
    """

    def __init__(self,c_phi_ini,v_phi_ini):
        self.c_phi_set = Reconstruction.from_ini(c_phi_ini)
        self.v_phi_set = Reconstruction.from_ini(v_phi_ini)
        if self.c_phi_set.fsky > 0.8:
            assert self.c_phi_set.fsky == self.v_phi_set.fsky
            extraname = f'_{self.c_phi_set.fsky:.2f}'
        else:
            extraname = ''
        fname = os.path.join(self.v_phi_set.lib_dir,f'n1{extraname}.pkl')
        if os.path.isfile(fname):
            print('Loading N1')
            self.n1 = pl.load(open(fname,'rb'))
        else:
            print('Calculating N1')
            self.n1 = self.get_n1()
            pl.dump(self.n1,open(fname,'wb'))

    def get_n1(self):
        """
        Calculate the N1 bias
        """
        n1 = self.c_phi_set.MCN0(self.c_phi_set.nsim) - self.v_phi_set.MCN0(self.v_phi_set.nsim)
        return n1
    
    def plot_n1(self):
        """
        Plot the N1 bias
        """
        plt.loglog(self.c_phi_set.cl_pp*self.c_phi_set.Lfac)
        plt.loglog(self.n1*self.c_phi_set.Lfac)
        plt.loglog(self.c_phi_set.norm*self.c_phi_set.Lfac)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-qlms', dest='qlms', action='store_true', help='reconsturction')
    parser.add_argument('-N0', dest='N0', action='store_true', help='MCN0')
    parser.add_argument('-RDN0', dest='RDN0', action='store_true', help='RDN0')
    parser.add_argument('-qlms_input', dest='qlms_input', action='store_true', help='Input Phi')
    parser.add_argument('-resp', dest='resp', action='store_true', help='response')
    parser.add_argument('-N1', dest='N1', action='store_true', help='N1')

    args = parser.parse_args()
    ini = args.inifile[0]

    if args.qlms:
        r = Reconstruction.from_ini(ini)
        r.job_phi()
    
    if args.N0:
        r = Reconstruction.from_ini(ini)
        r.job_MCN0()
    
    if args.RDN0:
        r = Reconstruction.from_ini(ini)
        r.job_RDN0()
    
    if args.qlms_input:
        r = Reconstruction.from_ini(ini)
        r.job_input_phi()

    if args.resp:
        r = Reconstruction.from_ini(ini)
        r.job_response()
    
    if args.N1:
        rc = f"{ini.split('.')[0]}_n1.ini"
        rv = ini
        n1 = N1(rc,rv)
        