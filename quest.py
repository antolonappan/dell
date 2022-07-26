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

    def __init__(self,filt_lib,Lmax,rlmin,rlmax,cl_unl,nbins):
        self.filt_lib = filt_lib
 

        self.Lmax = Lmax
        self.rlmin = rlmin
        self.rlmax = rlmax
        self.lib_dir = os.path.join(self.filt_lib.sim_lib.outfolder,'Reconstruction')
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
        self.binner = binning.multipole_binning(self.nbins,lmin=2,lmax=self.Lmax)
        self.B = self.binner.bc
        self.Bfac = (self.B*(self.B+1.))**2/(2*np.pi)
        
    
    def bin_cell(self,arr):
        return binning.binning(arr,self.binner)

    @classmethod
    def from_ini(cls,ini_file):
        filt_lib = Filtering.from_ini(ini_file)
        config = toml.load(ini_file)
        rc = config['Reconstruction']
        Lmax = rc['phi_lmax']
        rlmin = rc['cmb_lmin']
        rlmax = rc['cmb_lmax']
        cl_unl = rc['cl_unl']
        nbins = rc['nbins']
        return cls(filt_lib,Lmax,rlmin,rlmax,cl_unl,nbins)

    @property
    def __observed_spectra__(self):
        ocl = self.cl_len.copy()
        nt,ne,nb = self.filt_lib.sim_lib.noise_spectra(500)
        bt,be,bb = self.filt_lib.sim_lib.beam_spectra(500)
        ocl[0,:]  += (nt[:self.Lmax+1]/self.Tcmb**2)/bt[:self.Lmax+1]**2
        ocl[1,:]  += (ne[:self.Lmax+1]/self.Tcmb**2)/be[:self.Lmax+1]**2
        ocl[2,:]  += (nb[:self.Lmax+1]/self.Tcmb**2)/bb[:self.Lmax+1]**2
        return ocl
    
    def test_obs_for_norm(self):
        obs = self.__observed_spectra__.copy()
        cmb,_,_ = self.filt_lib.sim_lib.get_cmb_alms(0)
        plt.figure(figsize=(8,8))
        plt.loglog(hp.alm2cl(cmb[1])/self.Tcmb**2)
        plt.loglog(hp.alm2cl(cmb[2])/self.Tcmb**2)
        plt.loglog(obs[1,:])
        plt.loglog(obs[2,:])
        plt.axhline(np.radians(2.16/60)**2 /self.Tcmb**2)


    @property
    def get_norm(self):
        ocl = self.__observed_spectra__
        Ag, Ac = cs.norm_quad.qeb('lens',self.Lmax,self.rlmin,
                                  self.rlmax,self.cl_len[1,:],
                                  ocl[1,:],ocl[2,:])
        del Ac
        return Ag

    def get_phi(self,idx):
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
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            phi = self.get_phi(i)
        mpi.barrier()
    
    def job_phi_cross(self):
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            phi = self.get_phi_cross(i)
        mpi.barrier()

    def mean_field(self):
        fname = os.path.join(self.lib_dir,f"MF_fsky_{self.fsky:.2f}_{hash_array(self.mf_array)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            arr = np.zeros((self.Lmax+1,self.Lmax+1),dtype=complex)
            for i in tqdm(self.mf_array,desc="Calculating Mean Field",unit='Simulation'):
                arr += self.get_phi(i)
            arr /= len(self.mf_array)
            pl.dump(arr,open(fname,'wb'))
            return arr
    
    def mean_field_cl(self):
        return cs.utils.alm2cl(self.Lmax,self.mean_field())/self.fsky

    def get_phi_cl(self,idx):
        if idx in self.mf_array:
            raise ValueError("Simulation already in mean field array")
        else:
            return cs.utils.alm2cl(self.Lmax,self.get_phi(idx))/self.fsky


    def get_input_phi_sim(self,idx):
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
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            phi = self.get_input_phi_sim(i)
        mpi.barrier()

    def get_input_phi_cl(self,idx):
        return cs.utils.alm2cl(self.Lmax,self.Lmax,self.get_input_phi_sim(idx))


    def get_cl_phi_inXout(self,idx):
        almi = self.get_input_phi_sim(idx)
        almo = self.get_phi(idx) - self.mean_field()
        return cs.utils.alm2cl(self.Lmax,almi,almo)/self.fsky
    
    def response(self,idx):
        fname = os.path.join(self.rp_dir,f"response_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            almi = self.get_input_phi_sim(idx)
            almo = self.get_phi(idx) - self.mean_field()
            r =  cs.utils.alm2cl(self.Lmax,almi,almo)/cs.utils.alm2cl(self.Lmax,almi)
            r[0] = 0
            r[1] = 0
            pl.dump(r,open(fname,'wb'))
            return r
    
    def response_mean(self):
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
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            Null = self.response(i)
        mpi.barrier()
    
    def get_qcl(self,idx):
        return self.get_phi_cl(idx)  - self.MCN0() - self.mean_field_cl()

    def get_qcl_wR(self,idx):
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
        return arr

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
        clpp = self.cl_unl['pp'][:self.Lmax+1]/self.Tcmb**2
        cltt = self.cl_unl['tt'][:self.Lmax+1]/self.Tcmb**2
        cltp = self.cl_unl['tp'][:self.Lmax+1]/self.Tcmb**2
        Plm = self.get_input_phi_sim(idx)
        Tlm = cs.utils.gauss2alm_const(self.Lmax,clpp,cltt,cltp,Plm)
        del Plm
        tmap = cs.utils.hp_alm2map(self.nside,self.Lmax,self.Lmax,Tlm[1])*self.mask
        del Tlm
        Tlm = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,tmap)
        Plm = self.get_qlm_sim(idx)/self.Tcmb
        return cs.utils.alm2cl(self.Lmax,Tlm,Plm)/self.fsky



    

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

