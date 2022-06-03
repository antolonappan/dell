import healpy as hp
import basic
import curvedsky as cs
import cmb
import numpy as np
from utils import camb_clfile,timing,hash_array,cli
import os
import mpi
import pickle as pl
from tqdm import tqdm
import toml
import matplotlib.pyplot as plt
import binning
import analysis as ana
import pymaster as nmt


class RecoBase:

    def __init__(self,lib_dir,lib_dir_ap,fwhm,nside,nlev_p,
                      maskpath,nsim,len_cl_file,unl_cl_file,
                      cmb_sim_dir=None,cmb_sim_prefix=None,
                      exp_sim_dir=None,exp_sim_prefix=None,
                      phi_sim_dir=None,phi_sim_prefix=None,
                      FG=False,Lmax=1024,nbin=100,ana_lmax=1024,
                      MF_imin=400,MF_imax=500,extra_mask=None,
                      which_bin='namaster',noise_spectra=None
                ):

        if FG:
            assert (exp_sim_dir != None) & (exp_sim_dir != None)
        else:
            assert (cmb_sim_prefix != None) & (cmb_sim_dir != None)

        self.lib_dir = lib_dir
        self.lib_dir_ap = lib_dir_ap
        self.mass_dir = os.path.join(lib_dir,f'MASS_{lib_dir_ap}')
        self.filt_dir = os.path.join(lib_dir,f'CINV_{lib_dir_ap}')
        self.map_dir = os.path.join(lib_dir,f'MAP_{lib_dir_ap}')
        if mpi.rank == 0:
            os.makedirs(self.mass_dir, exist_ok=True)
            os.makedirs(self.filt_dir, exist_ok=True)
            os.makedirs(self.map_dir, exist_ok=True)
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
        self.nsim = nsim
        self.fsky = np.mean(self.mask)
        self.beam = 1./cmb.beam(self.fwhm,self.lmax)
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
        self.mf_array = np.arange(MF_imin,MF_imax)
        if extra_mask is not None:
            self.extra_mask = hp.ud_grade(hp.read_map(extra_mask),self.nside)
            print(f"An extra Mask is applied to data. Previous fsky is {self.fsky:.2f} and new fsky = {np.mean(self.extra_mask):.2f}")
            self.fsky = np.mean(self.extra_mask)
        else:
            self.extra_mask = np.ones_like(self.mask)


        self.nbin = nbin
        self.which_bin = which_bin
        if which_bin == 'cmblensplus':
            self.mb = binning.multipole_binning(self.nbin,lmin=2,lmax=ana_lmax)
            self.B = self.mb.bc
        elif which_bin == 'namaster':
            self.mb = nmt.bins.NmtBin.from_lmax_linear(self.Lmax,self.nbin)
            self.B = self.mb.get_effective_ells()
        else:
            raise ValueError
        if noise_spectra is None:
            self.NL = 0
            invn = self.mask *self.extra_mask * (np.radians(self.sigma/60)/self.Tcmb)**-2
            self.invN = np.reshape(np.array((invn,invn)),(2,1,self.npix))
        else:
            ne, nb = pl.load(open(os.path.join(self.lib_dir,noise_spectra),'rb'))
            ne, nb = ne[:self.lmax+1]*self.beam**2, nb[:self.lmax+1]*self.beam**2
            ne /= self.Tcmb**2
            nb /= self.Tcmb**2

            self.NL = np.reshape(np.array((cli(ne[:self.lmax+1]),cli(nb[:self.lmax+1]))),(2,1,self.lmax+1))
            invn = self.mask*self.extra_mask
            self.invN = np.reshape(np.array((invn,invn)),(2,1,self.npix))



    @classmethod
    def from_ini(cls,ini):
        config = toml.load(ini)
        fc = config['Folder']
        cc = config['CAMB']
        mc = config['Map']
        rc = config['Reconstruction']
        ac = config['Analysis']

        lib_dir = fc['lib_dir']
        lib_dir_a = fc['lib_dir_append']
        cmb_sim_dir = None if len(fc['cmb_dir']) == 0 else fc['cmb_dir']
        cmb_sim_prefix = None if len(fc['cmb_prefix']) == 0 else fc['cmb_prefix']
        exp_sim_dir = None if len(fc['exp_dir']) == 0 else fc['exp_dir']
        exp_sim_prefix = None if len(fc['exp_prefix']) == 0 else fc['exp_prefix']
        phi_sim_dir = None if len(fc['phi_dir']) == 0 else fc['phi_dir']
        phi_sim_prefix = None if len(fc['phi_prefix']) == 0 else fc['phi_prefix']

        fwhm = float(mc['fwhm'])
        nside = int(mc['nside'])
        nlev_p = float(mc['nlev_p'])
        maskpath = mc['maskpath']
        nsim = mc['nsim']
        len_cl_file = cc['cl_len']
        unl_cl_file = cc['cl_unl']
        FG = bool(mc['FG'])

        Lmax = rc['Lmax']

        ana_lmax = ac['lmax']
        nbin = ac['nbin']
        MF_imin = int(rc['MF_imin'])
        MF_imax = int(rc['MF_imax'])
        extra_mask = None if len(rc['mask'])==0 else rc['mask']
        which_bin = ac['which_bin']
        noise_spectra = None if len(rc['noise_spectra']) == 0 else rc['noise_spectra']

        return cls(lib_dir,lib_dir_a,fwhm,nside,nlev_p,maskpath,
                   nsim,len_cl_file,unl_cl_file,cmb_sim_dir,
                   cmb_sim_prefix,exp_sim_dir,exp_sim_prefix,
                   phi_sim_dir,phi_sim_prefix,
                   FG,Lmax,nbin,ana_lmax,MF_imin,MF_imax,extra_mask,
                   which_bin,noise_spectra)

    def bin_cell(self,arr):
        if self.which_bin == 'cmblensplus':
            return binning.binning(arr,self.mb)
        else:
            return self.mb.bin_cell(arr)

    @property
    def Lfac(self):
        L = self.L
        return (L*(L+1.))**2/(2*np.pi)
    @property
    def Bfac(self):
        L = self.B
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
        print(f"Maps without FG")
        fname = os.path.join(self.map_dir,f"exp_sim_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            Tlm, Elm, Blm = self.get_cmb_sim(idx)
            Tlm_f = hp.almxfl(Tlm/self.Tcmb,self.beam) + hp.synalm(self.Nl,lmax=self.lmax)
            Elm_f = hp.almxfl(Elm/self.Tcmb,self.beam)+ hp.synalm(self.Nl,lmax=self.lmax)
            Blm_f = hp.almxfl(Blm/self.Tcmb,self.beam) + hp.synalm(self.Nl,lmax=self.lmax)
            del (Tlm, Elm, Blm)
            T,Q,U = hp.alm2map([Tlm_f,Elm_f,Blm_f],self.nside)
            del T
            QU = np.reshape(np.array((Q*self.mask,U*self.mask)),(2,1,self.npix))
            pl.dump(QU, open(fname,'wb'))
            return QU

    def get_exp_sim(self,idx):
        print(f"Maps with FG")
        fname_main = os.path.join(self.map_dir,f"exp_sim_{idx:04d}.pkl")
        if os.path.isfile(fname_main):
            return pl.load(open(fname_main,'rb'))
        else:
            fname = os.path.join(self.exp_sim_dir,f"{self.exp_sim_pre}{idx:04d}.fits")
            #Tlm,Elm,Blm = hp.map2alm(hp.read_map(fname,(0,1,2)),lmax=self.lmax)
            T,Q,U = hp.alm2map(hp.read_alm(fname,(1,2,3))/self.Tcmb,self.nside)
            del T
            QU = np.reshape(np.array((Q,U)),(2,1,self.npix))
            pl.dump(QU, open(fname_main,'wb'))
            return QU

    def get_sim(self,idx):
        return self.get_exp_sim(idx) if self.FG else self.make_exp_sim(idx)

    def get_falm_sim(self,idx,filt='',ret=None):
        fname = os.path.join(self.filt_dir,f"cinv_sim_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            E,B = pl.load(open(fname,'rb'))
        else:
            Bl = np.reshape(self.beam,(1,self.lmax+1))
            QU = self.get_sim(idx)*self.extra_mask
            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:],
                                         Bl,self.invN,QU,chn=1,itns=[1000],eps=[1e-5],
                                         filter=filt,ro=10,inl=self.NL)
            pl.dump((E,B),open(fname,'wb'))

        if ret is None:
            return E, B
        elif ret == 'E':
            del B
            return E
        elif ret == 'B':
            del E
            return B
        else:
            raise ValueError

#    @timing
    def get_qlm_sim(self,idx):
        fname = os.path.join(self.mass_dir,f"phi_sim_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E,B = self.get_falm_sim(idx)
            glm, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,self.cl_len[1,:],E,B)
            del(clm)
            glm *= self.norm[:,None]
            pl.dump(glm,open(fname,'wb'))
            return glm

    def get_qlm_cross_sim(self,idx):
        assert idx < self.nsim - 1
        fname = os.path.join(self.mass_dir,f"phi_cross_sim_fsky_{self.fsky:.2f}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            E = self.get_falm_sim(idx,ret="E")
            B = self.get_falm_sim(idx+1,ret="B")
            glm, clm = cs.rec_lens.qeb(self.Lmax,self.rlmin,self.rlmax,self.cl_len[1,:],E,B)
            del clm
            glm *= self.norm[:,None]
            pl.dump(glm,open(fname,'wb'))
            return glm

    def get_qcl_cross_sim(self,idx):
        return cs.utils.alm2cl(self.Lmax,self.get_qlm_cross_sim(idx))

    def get_qcl_cross_mean(self,n):
        m = np.zeros_like(self.L)
        for i in tqdm(range(n), desc='corss spectra stat',unit='simulation'):
            m += self.get_qcl_cross_sim(i)
        return m/n

    def get_qcl_sim(self,idx):
        if idx in self.mf_array:
            raise ValueError
        return cs.utils.alm2cl(self.Lmax,self.get_qlm_sim(idx)-self.mean_field())/self.fsky - self.MCN0()

    def get_qcl_stat(self,n,ret='dl'):
        if ret == 'cl':
            lfac = 1.0
        elif ret == 'dl':
            lfac = self.Lfac
        else:
            raise ValueError
        cl = []
        for i in tqdm(range(n), desc='qcl stat',unit='simulation'):
            cl.append(self.bin_cell(lfac*self.get_qcl_sim(i)))
        return np.array(cl)


    def get_kappa_alm_sim(self,idx):
        fl = self.L * (self.L + 1)/2
        return cs.utils.almxfl(self.Lmax,self.Lmax,self.get_qlm_sim(idx),fl)

    def get_kappa_map_sim(self,idx):
        return cs.utils.hp_alm2map(self.nside,self.Lmax,
                                   self.Lmax,self.get_kappa_alm_sim(idx))

    def run_job(self,nsim):
        jobs = np.arange(nsim)
        for i in jobs[mpi.rank::mpi.size]:
            Null = self.get_qlm_sim(i)

    def mean_field(self):
        fname = os.path.join(self.mass_dir,f"MF_fsky_{self.fsky:.2f}_{hash_array(self.mf_array)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            arr = np.zeros((self.Lmax+1,self.Lmax+1),dtype=complex)
            for i in tqdm(self.mf_array,desc="Calculating Mean Field",unit='Simulation'):
                arr += self.get_qlm_sim(i)
            arr /= len(self.mf_array)
            pl.dump(arr,open(fname,'wb'))
            return arr

    def mean_field_cl(self):
        return cs.utils.alm2cl(self.Lmax,self.mean_field())/self.fsky

    def MCN0(self,n=300):
        fname = os.path.join(self.mass_dir,f"MCN0_{n}_fsky_{self.fsky:.2f}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            arr = self.get_qcl_cross_mean(n)/self.fsky
            pl.dump(arr,open(fname,'wb'))
            return arr

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


    def SNR_phi(self,n):
        cl_pp = self.get_qcl_stat(n,ret='cl')
        stat = ana.statistics(ocl=1.,scl=cl_pp)
        stat.get_amp(fcl=cl_pp.mean(axis=0))
        return 1/stat.sA


    def get_tXphi(self,idx):
        clpp = self.cl_unl['pp'][:self.Lmax+1]/self.Tcmb**2
        cltt = self.cl_unl['tt'][:self.Lmax+1]/self.Tcmb**2
        cltp = self.cl_unl['tp'][:self.Lmax+1]/self.Tcmb**2
        Plm = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,
                            hp.alm2map(self.__get_input_phi_sim__(idx),
                                       self.nside)/self.Tcmb)
        Tlm = cs.utils.gauss2alm_const(self.Lmax,clpp,cltt,cltp,Plm)
        del Plm
        tmap = cs.utils.hp_alm2map(self.nside,self.Lmax,self.Lmax,Tlm[1])*self.mask
        del Tlm
        Tlm = cs.utils.hp_map2alm(self.nside,self.Lmax,self.Lmax,tmap)
        Plm = self.get_qlm_sim(idx)/self.Tcmb
        return cs.utils.alm2cl(self.Lmax,Tlm,Plm)/self.fsky

    def tXphi_stat(self,n):
        fname = os.path.join(self.mass_dir,f"tXphi_{n}_{hash_array(self.B)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            cl = []
            for i in tqdm(range(n), desc='Calculating TempXphi',unit='simulation'):
                cl.append(self.bin_cell(self.get_tXphi(i)))
            cl = np.array(cl)
            pl.dump(cl,open(fname,'wb'))
            return cl

    def plot_tXphi_stat(self,n):
        cl = self.tXphi_stat(n)
        plt.errorbar(self.B,cl.mean(axis=0),yerr=cl.std(axis=0))
        plt.loglog(self.L,self.cl_unl['tp'][:self.Lmax+1]/self.Tcmb**2)

    def SNR_tp(self,n):
        cltp = self.tXphi_stat(n)[:,:]
        stat = ana.statistics(ocl=1.,scl=cltp)
        stat.get_amp(fcl=cltp.mean(axis=0))
        return 1/stat.sA




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-qlms', dest='qlms', action='store_true', help='reconstruct')
    parser.add_argument('-map', dest='map', action='store_true', help='map')
    args = parser.parse_args()
    ini = args.inifile[0]

    if args.qlms:
        r = RecoBase.from_ini(ini)
        r.run_job(500)

    if args.map:
        r = RecoBase.from_ini(ini)
        jobs = np.arange(500)
        for i in jobs[mpi.rank::mpi.size]:
            Null = r.get_sim(i)