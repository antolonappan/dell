import healpy as hp
import basic
import curvedsky as cs
import cmb
import numpy as np
from utils import camb_clfile,timing,hash_array
import os
import mpi
import pickle as pl
from tqdm import tqdm
import toml
import matplotlib.pyplot as plt
#import pymaster as nmt
import binning


class RecoBase:

    def __init__(self,lib_dir,fwhm,nside,nlev_p,maskpath,
                      len_cl_file,unl_cl_file,
                      cmb_sim_dir=None,cmb_sim_prefix=None,
                      exp_sim_dir=None,exp_sim_prefix=None,
                      phi_sim_dir=None,phi_sim_prefix=None,
                      FG=False,
                      Lmax=1024,
                      nbin=100,
                ):

        if FG:
            assert (exp_sim_dir != None) & (exp_sim_dir != None)
        else:
            assert (cmb_sim_prefix != None) & (cmb_sim_dir != None)

        self.lib_dir = lib_dir
        self.mass_dir = os.path.join(lib_dir,'MASS')
        if mpi.rank == 0:
            os.makedirs(self.mass_dir, exist_ok=True)
        mpi.barrier()

        self.fwhm = fwhm
        self.nside = nside         
        self.Tcmb  = 2.726e6
        self.lmax  = 2*nside     
        self.npix  = 12*nside**2
        self.l = np.linspace(0,self.lmax,self.lmax+1)
        self.sigma = nlev_p      
        self.Nl = (self.sigma*(np.pi/10800.)/self.Tcmb)**2 * np.ones(self.lmax+1)
        self.Lmax  = Lmax
        self.rlmin, self.rlmax = 200, 1024 
        self.L = np.linspace(0,self.Lmax,self.Lmax+1)
        self.mask = hp.ud_grade(hp.read_map(maskpath),self.nside)
        self.fsky = np.mean(self.mask)
        self.beam = 1./cmb.beam(self.fwhm,self.lmax)
        invn = self.mask * (np.radians(self.sigma/60)/self.Tcmb)**-2
        self.invN = np.reshape(np.array((invn,invn)),(2,1,self.npix))
        self.cl_len = cmb.read_camb_cls(len_cl_file,ftype='lens',output='array')[:,:self.lmax+1]
        self.cl_unl = camb_clfile(unl_cl_file)
        self.norm = self.get_norm
        self.cmb_sim_dir = cmb_sim_dir
        self.cmb_sim_pre = cmb_sim_prefix
        self.exp_sim_dir = exp_sim_dir
        self.exp_sim_pre = exp_sim_prefix
        self.phi_sim_dir = phi_sim_dir
        self.phi_sim_pre = phi_sim_prefix
        self.FG = FG
        self.nbin = nbin
        self.mb = binning.multipole_binning(self.nbin,lmin=2,lmax=self.Lmax)
        #self.bin = nmt.bins.NmtBin.from_lmax_linear(self.Lmax,10)
        #self.B = self.bin.get_effective_ells()

    @property
    def Lfac(self):
        L = self.L
        return (L*(L+1.))**2/(2*np.pi)

    @property
    def __observed_spectra__(self):
        ocl = self.cl_len.copy()
        ocl[1,:]  += self.Nl/self.beam**2
        ocl[2,:]  += self.Nl/self.beam**2
        return ocl

    @property
    def get_norm(self):
        ocl = self.__observed_spectra__
        Ag, Ac = cs.norm_quad.qeb('lens',self.Lmax,self.rlmin,
                                  self.rlmax,self.cl_len[1,:],
                                  ocl[1,:],ocl[2,:])
        del Ac
        return Ag

    def get_cmb_sim(self,idx):
        fname = os.path.join(self.cmb_sim_dir,f"{self.cmb_sim_pre}{idx:04d}.fits")
        return hp.map2alm(hp.read_map(fname,(0,1,2)),lmax=self.lmax)

    def make_exp_sim(self,idx):
        Tlm, Elm, Blm = self.get_cmb_sim(idx)
        Tlm_f = hp.almxfl(Tlm/self.Tcmb,self.beam) + hp.synalm(self.Nl,lmax=self.lmax)
        Elm_f = hp.almxfl(Elm/self.Tcmb,self.beam)+ hp.synalm(self.Nl,lmax=self.lmax)
        Blm_f = hp.almxfl(Blm/self.Tcmb,self.beam) + hp.synalm(self.Nl,lmax=self.lmax)
        del (Tlm, Elm, Blm)
        T,Q,U = hp.alm2map([Tlm_f,Elm_f,Blm_f],self.nside)
        del T
        return np.reshape(np.array((Q*self.mask,U*self.mask)),(2,1,self.npix))

    def get_exp_sim(self,idx):
        fname = os.path.join(self.exp_sim_dir,f"{self.exp_sim_pre}{idx:04d}.fits")
        Tlm,Elm,Blm = hp.map2alm(hp.read_map(fname,(0,1,2)),lmax=self.lmax)
        T,Q,U = hp.alm2map([Tlm_f,Elm_f,Blm_f],self.nside)
        del (Tlm,Elm,Blm,T)
        return np.reshape(np.array((Q,U)),(2,1,self.npix))

    def get_sim(self,idx):
        return self.get_exp_sim(idx) if self.FG else self.make_exp_sim(idx)

    def get_falm_sim(self,idx):
        Bl = np.reshape(self.beam,(1,self.lmax+1))
        QU = self.get_sim(idx)
        E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:],
                                     Bl,self.invN,QU,chn=1,itns=[1000],eps=[1e-5],
                                     filter='',ro=10)
        return E, B

    def get_qlm_sim(self,idx):
        fname = os.path.join(self.mass_dir,f"phi_sim_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E,B = self.get_falm_sim(idx)
            glm, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,self.cl_len[1,:],E,B)
            del(clm)
            glm *= self.norm[:,None]
            pl.dump(glm,open(fname,'wb'))
            return glm

    def get_qcl_sim(self,idx):
        return cs.utils.alm2cl(self.Lmax,self.get_qlm_sim(idx))

    def get_cross_qcl_sim(self,idx1,idx2):
        return cs.utils.alms2cl(self.Lmax,
                                self.get_qlm_sim(idx1),
                                self.get_qlm_sim(idx2))

    def get_kappa_alm_sim(self,idx):
        fl = self.L * (self.L + 1)/2
        return cs.utils.almxfl(self.Lmax,self.Lmax,self.get_qlm_sim(idx),fl)

    def get_kappa_map_sim(self,idx):
        return cs.utils.hp_alm2map(self.nside,self.Lmax,
                                   self.Lmax,self.get_kappa_alm_sim(idx))

    def run_job(self,nsim):
        jobs = np.arange(nsim)
        for i in jobs[mpi.rank::mpi.size]:
            Null = self.get_qcl_sim(i)

    def mean_field(self,idx_array):
        fname = os.path.join(self.lib_dir,f"MF_{hash_array(idx_array)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            arr = np.zeros((self.Lmax+1,self.Lmax+1),dtype=complex)
            for i in tqdm(idx_array,desc="Calculating Mean Field",unit='Simulation'):
                arr += self.get_qlm_sim(i)
            arr /= len(idx_array)
            pl.dump(arr,open(fname,'wb'))
            return arr

    def mean_field_cl(self,idx_array):
        return cs.utils.alm2cl(self.Lmax,self.mean_field(idx_array))

    def plot_recon_sim(self,idx):
        theory = self.cl_unl['pp'][:self.Lmax+1]
        plt.figure(figsize=(8,8))
        plt.loglog(self.L,self.Lfac*self.get_qcl_sim(idx)/self.fsky)
        plt.loglog(self.L,self.Lfac*(theory+self.norm))

    def plot_mf(self,idx,idxe=None):
        theory = self.cl_unl['pp'][:self.Lmax+1]
        plt.figure(figsize=(8,8))
        plt.loglog(self.L,self.Lfac*theory)
        plt.loglog(self.L,self.Lfac*self.mean_field_cl(idx))
        if idxe is not None:
            plt.loglog(self.L,self.Lfac*self.mean_field_cl(idxe))

    def __get_input_phi_sim__(self,idx):
        fname = os.path.join(self.phi_sim_dir,f"{self.phi_sim_pre}{idx:04d}.fits")
        return hp.read_alm(fname)

    def get_kappa_alm_inp_sim(self,idx):
        fl = self.L * (self.L + 1)/2
        return hp.almxfl(self.__get_input_phi_sim__(idx),fl)

    def get_kappa_map_inp_sim(self,idx):
        return hp.alm2map(self.get_kappa_alm_inp_sim(idx),nside=self.nside)*self.mask

    def get_input_phi_sim(self,idx):
        fl = 2/(self.L * (self.L + 1))
        klm = hp.map2alm(self.get_kappa_map_inp_sim(idx))
        hp.almxfl(klm,fl,inplace=True)
        klm[0] = 0
        return klm

    def get_cl_phi_inXout(self,idx):
        almi = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,hp.alm2map(self.get_input_phi_sim(idx),self.nside))
        almo = self.get_qlm_sim(idx)
        
        return cs.utils.alm2cl(self.Lmax,almi,almo)/self.fsky

    def plot_input_sim(self,idx):
        theory = self.cl_unl['pp'][:self.Lmax+1]
        plt.figure(figsize=(8,8))
        plt.loglog(self.L,self.Lfac*theory)
        almi = self.get_input_phi_sim(idx)
        plt.loglog(self.L,self.Lfac*hp.alm2cl(almi,lmax_out=self.Lmax)/self.fsky)

    def plot_inXout(self,idx):
        theory = self.cl_unl['pp'][:self.Lmax+1]
        plt.figure(figsize=(8,8))
        plt.loglog(self.L,self.Lfac*theory)  
        plt.loglog(self.L,self.Lfac*self.get_cl_phi_inXout(idx))

    def input_stat(self,n=100):
        
        #fname = os.path.join(self.lib_dir,f"fid_stat_{n}_{hash_array(self.B)}.pkl")
        fname = os.path.join(self.lib_dir,'fid_stat_100_ac154d7d78bb43255d7aa1587ca0ab9b0520f0b948be91c0bb8699b7.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            arr = []
            for i in tqdm(range(n),desc='Calcualting fiducial stat',unit='realisation'):
                cl = hp.alm2cl(self.__get_input_phi_sim__(i),lmax_out=self.Lmax)
                #arr.append(self.bin.bin_cell(cl))
                arr.append(binning.binning(cl,self.mb))
            arr = np.array(arr)

            stat = {}
            stat['mean'] = arr.mean(axis=0)
            stat['cov'] = np.cov(arr.T)
            del arr
            pl.dump(stat,open(fname,'wb'))
            return stat

    def plot_qcl_stat(self,n=100):
        cl = []
        for idx in tqdm(range(100),desc='Calculating reconstruction stat',unit='realisation'):
            cl.append(self.bin.bin_cell(self.Lfac*((self.get_qcl_sim(idx)/self.fsky)-self.norm)))

        cl = np.array(cl)
        theory = self.cl_unl['pp'][:self.Lmax+1]
        plt.figure(figsize=(8,8))
        plt.loglog(self.L,self.Lfac*theory)
        plt.errorbar(self.B,cl.mean(axis=0),yerr=cl.std(axis=0),fmt='o')

    def SNR(self,n=100):
        stat = self.input_stat(n)
        input_mean = stat['mean']
        input_cov = stat['cov']
        inv_cov = np.linalg.inv(input_cov)

        a_b = input_mean * np.dot(inv_cov,input_mean)

        select = np.where((self.B > 20) & (self.B < 200))[0]


        al_phi = []
        for idx in tqdm(range(n),desc='Calculating reconstruction stat',unit='realisation'):
            output_cl = self.bin.bin_cell((self.get_qcl_sim(idx)/self.fsky)-self.norm)
            A_b = output_cl/input_mean
            al_phi.append(np.sum(a_b[select]*A_b[select])/a_b[select].sum())
        al_phi = np.array(al_phi)
        return 1/np.std(al_phi)

    def response(self,idx):
        return self.get_cl_phi_inXout(idx)/self.cl_unl['pp'][:self.Lmax+1]
    
    def plot_var(self,n=100):
        output_cl = []
        for idx in tqdm(range(n),desc='Calculating var',unit='realisation'):
            output_cl.append(self.bin.bin_cell((self.get_qcl_sim(idx)/self.fsky)))
        sim_var = np.var(np.array(output_cl),axis=0)
        
        cl_pp = self.bin.bin_cell(self.cl_unl['pp'][:self.Lmax+1] + self.norm)
        f = (2*self.fsky)/(((2*self.B) + 1)*10)
        tru_var = f * cl_pp**2
        
        plt.loglog(self.B,tru_var,label='anal')
        plt.loglog(self.B,sim_var,label='sim')
        plt.legend()





class RecoIni(RecoBase):

    def __init__(self,ini):
        config = toml.load(ini)
        fc = config['Folder']
        cc = config['CAMB']
        mc = config['Map']
        
        lib_dir = fc['lib_dir']
        cmb_sim_dir = None if len(fc['cmb_dir']) == 0 else fc['cmb_dir']
        cmb_sim_prefix = None if len(fc['cmb_prefix']) == 0 else fc['cmb_prefix']
        exp_sim_dir = None if len(fc['exp_dir']) == 0 else fc['exp_dir']
        exp_sim_prefix = None if len(fc['cmb_prefix']) == 0 else fc['exp_prefix']
        phi_sim_dir = None if len(fc['phi_dir']) == 0 else fc['phi_dir']
        phi_sim_prefix = None if len(fc['phi_prefix']) == 0 else fc['phi_prefix']        
        
        fwhm = float(mc['fwhm'])
        nside = int(mc['nside'])
        nlev_p = float(mc['nlev_p'])
        maskpath = mc['maskpath']
        len_cl_file = cc['cl_len']
        unl_cl_file = cc['cl_unl']
        FG = bool(mc['FG'])
    
        
        super().__init__(lib_dir,fwhm,nside,nlev_p,maskpath,
                      len_cl_file,unl_cl_file,
                      cmb_sim_dir,cmb_sim_prefix,
                      exp_sim_dir,exp_sim_prefix,
                      phi_sim_dir,phi_sim_prefix,
                      FG,
                      Lmax,
                      nbin,
                      )



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-qlms', dest='qlms', action='store_true', help='reconstruct')
    args = parser.parse_args()
    ini = args.inifile[0]

    if args.qlms:
        r = RecoIni(ini)
        r.run_job(500)