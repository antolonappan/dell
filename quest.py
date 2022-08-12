from cProfile import label
import os
import healpy as hp
import mpi
import numpy as np
import matplotlib.pyplot as plt
import curvedsky as cs
import cmb
from utils import camb_clfile,cli,hash_array
import pickle as pl
from filtering import Filtering
import toml
from tqdm import tqdm
import analysis as ana
import binning


class Reconstruction:
    """
    Class to reconstruct the lensing potentials from the filtered CMB fields.

    filt_lib: class object : filtering.Filtering 
    Lmax: int : maximum multipole of the reconstruction
    rlmin: int : minimum multipole of CMB for the reconstruction
    rlmax: int : maximum multipole of CMB for the reconstruction
    cl_unl: str : path to the unlensed CMB power spectrum
    nbins: int : number of bins for the multipole binning
    """

    def __init__(self,filt_lib,Lmax,rlmin,rlmax,cl_unl,nbins,tp_nbins,N1_file):
        self.filt_lib = filt_lib
 

        self.Lmax = Lmax
        self.rlmin = rlmin
        self.rlmax = rlmax
        self.lib_dir = os.path.join(self.filt_lib.sim_lib.outfolder,
                                    f'Reconstruction_{self.rlmin}_{self.rlmax}')
        self.in_dir = os.path.join(self.lib_dir,'input')
        self.plm_dir = os.path.join(self.lib_dir,'plm')
        self.px_dir = os.path.join(self.lib_dir,'px')
        self.rp_dir = os.path.join(self.lib_dir,'response')
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
            os.makedirs(self.in_dir,exist_ok=True)
            os.makedirs(self.plm_dir,exist_ok=True)
            os.makedirs(self.px_dir,exist_ok=True)
            os.makedirs(self.rp_dir,exist_ok=True)
        
        self.mask = self.filt_lib.mask
        self.fsky = self.filt_lib.fsky
        self.nside = self.filt_lib.nside
        self.cl_len = self.filt_lib.cl_len[:,:self.Lmax+1]
        self.cl_pp = camb_clfile(cl_unl)['pp'][:self.Lmax+1]
        self.cl_unl = camb_clfile(cl_unl)
        self.beam = self.filt_lib.beam[:self.Lmax+1]
        self.Tcmb = self.filt_lib.Tcmb
        self.nsim = self.filt_lib.nsim

        self.norm = self.get_norm

        self.L = np.arange(self.Lmax+1)
        self.Lfac = (self.L*(self.L+1.))**2/(2*np.pi)
        self.mf_array = np.arange(400,500)

        self.nbins = nbins
        self.tp_nbins = tp_nbins
        self.binner = binning.multipole_binning(self.nbins,lmin=2,lmax=self.Lmax)
        self.binnertp = binning.multipole_binning(self.tp_nbins,lmin=2,lmax=self.Lmax)
        self.B = self.binner.bc
        self.Btp = self.binnertp.bc
        self.Bfac = (self.B*(self.B+1.))**2/(2*np.pi)
        self.N1 = pl.load(open(N1_file,'rb')) if os.path.isfile(N1_file) else np.zeros(self.Lmax+1)
        self.N1[:50] = 0
    def bin_cell(self,arr):
        """
        binning function for the multipole bins
        """
        return binning.binning(arr,self.binner)
    
    def bin_cell_tp(self,arr):
        """
        binning function for the multipole bins
        """
        return binning.binning(arr,self.binnertp)

    @classmethod
    def from_ini(cls,ini_file):
        """
        Load the reconstruction object from a ini file."""
        filt_lib = Filtering.from_ini(ini_file)
        config = toml.load(ini_file)
        rc = config['Reconstruction']
        Lmax = rc['phi_lmax']
        rlmin = rc['cmb_lmin']
        rlmax = rc['cmb_lmax']
        cl_unl = rc['cl_unl']
        nbins = rc['nbins']
        tp_nbins = rc['nbins_tp']
        N1_file = rc['N1']
        return cls(filt_lib,Lmax,rlmin,rlmax,cl_unl,nbins,tp_nbins,N1_file)

    @property
    def __observed_spectra__(self):
        """
        Calculate the expected observed spectra using ILC noise
        and effective beam
        """
        fg = False
        cmb = False
        ocl = self.cl_len.copy()
        nt,ne,nb = self.filt_lib.sim_lib.noise_spectra(self.filt_lib.sim_lib.nsim)
        bt,be,bb = self.filt_lib.sim_lib.beam_spectra(self.filt_lib.sim_lib.nsim)
        #(nb[:self.Lmax+1]/self.Tcmb**2)/bb[:self.Lmax+1]**2
        if fg:
            print('fg_res included in response')
            ft,fe,fb = self.filt_lib.sim_lib.fg_res_mean(500)
            ocl[0,:] += ft[:self.Lmax+1]*bt[:self.Lmax+1]**2
            ocl[1,:] += fe[:self.Lmax+1]*be[:self.Lmax+1]**2
            ocl[2,:] += fb[:self.Lmax+1]*bb[:self.Lmax+1]**2

        ocl[0,:] += nt[:self.Lmax+1]/bt[:self.Lmax+1]**2
        ocl[1,:] += ne[:self.Lmax+1]/be[:self.Lmax+1]**2
        ocl[2,:] += nb[:self.Lmax+1]/bb[:self.Lmax+1]**2

        if cmb:
            print('cmb included in response')
            ctt,cee,cbb = self.filt_lib.sim_lib.cmb_mean(500)
            ocl = np.zeros_like(ocl)
            ocl[0,:] += ctt[:self.Lmax+1]
            ocl[1,:] += cee[:self.Lmax+1]
            ocl[2,:] += cbb[:self.Lmax+1]
        return ocl



    def test_obs_for_norm(self):
        """
        Test the observed spectra for the normalization is visually acceptable.
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
        """
        fname = os.path.join(self.plm_dir,f"phi_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E,B = self.filt_lib.cinv_EB(idx)
            glm, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                       self.cl_len[1,:self.Lmax+1],
                                       E[:self.Lmax+1,:self.Lmax+1],
                                       B[:self.Lmax+1,:self.Lmax+1])
            del(clm)
            glm *= self.norm[:,None]
            pl.dump(glm,open(fname,'wb'))
            return glm

    def get_phi_cross(self,idx):
        """
        Reconstruct the potential using filtered Fields with different CMB fields
        If E modes is from ith simulation then B modes is from (i+1)th simulation
        """
        myidx = np.pad(np.arange(self.nsim),(0,1),'constant',constant_values=(0,0))
        fname = os.path.join(self.px_dir,f"phi_cross_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E,_ = self.filt_lib.cinv_EB(myidx[idx])
            _,B = self.filt_lib.cinv_EB(myidx[idx+1])
            glm, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,
                                       self.cl_len[1,:self.Lmax+1],
                                       E[:self.Lmax+1,:self.Lmax+1],
                                       B[:self.Lmax+1,:self.Lmax+1])
            del clm
            glm *= self.norm[:,None]
            pl.dump(glm,open(fname,'wb'))
            return glm
    
    def job_phi(self):
        """
        MPI job for the potential reconstruction.
        """
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            phi = self.get_phi(i)
        mpi.barrier()
    
    def job_phi_cross(self):
        """
        MPI job for the potential reconstruction with different CMB fields.
        """
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            phi = self.get_phi_cross(i)
        mpi.barrier()

    def mean_field(self):
        """
        Calcualte the mean field.
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

    def __kfac__(self):
        nhl = self.MCN0()
        fl = self.cl_pp/(self.cl_pp+ nhl )
        fl[0] = 0
        fl[1] = 0
        return fl

    def wf_phi(self,idx):
        phi = self.get_phi(idx) - self.mean_field()
        return cs.utils.almxfl(self.Lmax,self.Lmax,phi,self.__kfac__())
    
    def deflection_angle(self,idx):
        """
        Calculate the deflection angle.
        """
        wfphi = self.wf_phi(idx)
        dl = np.sqrt(np.arange(self.Lmax + 1, dtype=float) * np.arange(1, self.Lmax + 2))
        return cs.utils.almxfl(self.Lmax,self.Lmax,wfphi,dl)



    
    
    
    def mean_field_cl(self):
        """
        Mean field cl
        """
        return cs.utils.alm2cl(self.Lmax,self.mean_field())/self.fsky

    def get_phi_cl(self,idx):
        """
        Get the cl of the potential.
        """
        if idx in self.mf_array:
            raise ValueError("Simulation already in mean field array")
        else:
            return cs.utils.alm2cl(self.Lmax,self.get_phi(idx))/self.fsky


    def get_input_phi_sim(self,idx):
        """
        Get the masked input potential alms
        """
        if self.nsim <200:
            print("input phi is constant")
            dir_ = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps_c/MASS"
            fname = os.path.join(dir_,f"phi_sims.fits")
        else:
            print("input phi is from variying")
            dir_ = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps/MASS"
            fname = os.path.join(dir_,f"phi_sims_{idx:04d}.fits")
        fnamet = os.path.join(self.in_dir,f"phi_sims_{idx:04d}.pkl")
        if os.path.isfile(fnamet):
            return pl.load(open(fnamet,'rb'))
        else:
            plm = hp.read_alm(fname)
            fl = self.L * (self.L + 1)/2
            klm = hp.almxfl(plm,fl)
            kmap = hp.alm2map(klm,nside=self.nside)*self.mask
            klm_n = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,kmap)
            plm_n = cs.utils.almxfl(self.Lmax,self.Lmax,klm_n,1/fl)
            pl.dump(plm_n,open(fnamet,'wb')) 
            return plm_n

    def job_input_phi(self):
        """
        MPI job for the input potential reconstruction.
        """
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            phi = self.get_input_phi_sim(i)
        mpi.barrier()

    def get_input_phi_cl(self,idx):
        """
        Get the cl of the input potential.
        """
        return cs.utils.alm2cl(self.Lmax,self.get_input_phi_sim(idx))


    def get_cl_phi_inXout(self,idx):
        """
        get input X output potential
        """

        almi = self.get_input_phi_sim(idx)
        almo = self.get_phi(idx) #- self.mean_field()
        return cs.utils.alm2cl(self.Lmax,almi,almo)/self.fsky
    
    def response(self,idx):
        """
        Calculate the responce

        r = cl^{cross} / cl^{input}
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


    def job_response(self):
        """
        MPI job for the response calculation.
        """
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            Null = self.response(i)
        mpi.barrier()
    
    def get_qcl(self,idx):
        """
        Get the cl_phi = cl_recon - N0 - mean_field
        """
        return self.get_phi_cl(idx)  - self.MCN0() - self.mean_field_cl() #- (self.N1*self.response_mean()**2)

    def get_qcl_wR(self,idx):
        """
        Get the cl_phi = (cl_recon - N0 - mean_field)/ response*82
        """
        return self.get_qcl(idx)/self.response_mean()**2
    
    def get_qcl_wR_stat(self,n=400,ret='dl'):


        if ret == 'cl':
            lfac = 1.0
        elif ret == 'dl':
            lfac = self.Lfac
        else:
            raise ValueError
        cl = []
        for i in tqdm(range(n), desc='qcl stat',unit='simulation'):
            cl.append(self.bin_cell(lfac*self.get_qcl_wR(i)))
        
        cl = np.array(cl)
        return cl   

    
    def MCN0(self,n=400):
        def get_qcl_cross_sim(idx):
            return cs.utils.alm2cl(self.Lmax,self.get_phi_cross(idx))/self.fsky

        def get_qcl_cross_mean(n):
            m = np.zeros(self.Lmax+1,dtype=np.float64)
            for i in tqdm(range(n), desc='corss spectra stat',unit='simulation'):
                m += get_qcl_cross_sim(i)
            return m/n
        
        fname = os.path.join(self.lib_dir,f"MCN0_{n}_fsky_{self.fsky:.2f}.pkl")
        if os.path.isfile(fname):
            arr = pl.load(open(fname,'rb'))
        else:
            arr = get_qcl_cross_mean(n)
            pl.dump(arr,open(fname,'wb'))

        arr  += (1/n) * (arr+self.cl_pp[:self.Lmax+1])
        return arr#/self.response_mean()**2

    def get_qcl_stat(self,n=400,ret='dl',recache=False):
        fname = os.path.join(self.lib_dir,f"qcl_stat{self.nbins}_{n}_fsky_{self.fsky:.2f}_{ret}.pkl")
        if os.path.isfile(fname) and (not recache):
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
                cl.append(self.bin_cell(lfac*self.get_qcl(i)))
            
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl
        

    def SNR_phi(self,n=400):
        cl_pp = self.get_qcl_stat(n,'cl')
        stat = ana.statistics(ocl=1.,scl=cl_pp)
        stat.get_amp(fcl=cl_pp.mean(axis=0))
        return 1/stat.sA

    def get_tXphi(self,idx):
        """
        Get the Cl_{temp, phi}
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
    
    def tXphi_stat(self,n):
        fname = os.path.join(self.lib_dir,f"tXphi_{n}_{hash_array(self.Btp)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            cl = []
            for i in tqdm(range(n), desc='Calculating TempXphi',unit='simulation'):
                cl.append(self.bin_cell_tp(self.get_tXphi(i)))
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl

    def plot_tXphi_stat(self,n):
        cl = self.tXphi_stat(n)
        plt.errorbar(self.B,cl.mean(axis=0),yerr=cl.std(axis=0))
        plt.plot(self.L,self.cl_unl['tp'][:self.Lmax+1])
        plt.semilogy()
        plt.xlim(2,100)

    def SNR_tp(self,n):
        cltp = self.tXphi_stat(n)[:,:]
        stat = ana.statistics(ocl=1.,scl=cltp)
        stat.get_amp(fcl=cltp.mean(axis=0))
        return 1/stat.sA



    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-qlms', dest='qlms', action='store_true', help='reconsturction')
    parser.add_argument('-qlms_cross', dest='qlms_cross', action='store_true', help='reconsturction')
    parser.add_argument('-qlms_input', dest='qlms_input', action='store_true', help='reconsturction')
    parser.add_argument('-resp', dest='resp', action='store_true', help='reconsturction')

    args = parser.parse_args()
    ini = args.inifile[0]

    r = Reconstruction.from_ini(ini)

    if args.qlms:
        r.job_phi()
    
    if args.qlms_cross:
        r.job_phi_cross()
    
    if args.qlms_input:
        r.job_input_phi()

    if args.resp:
        r.job_response()

