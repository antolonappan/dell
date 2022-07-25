import curvedsky as cs
import mpi
import numpy as np
from workspace.LBlens.old_reco import RecoBase
import os
import toml
import pickle as pl
from utils import cli



class Delens:

    def __init__(self,recon):
        self.recon = recon
        self.lib_dir = os.path.join(self.recon.lib_dir,f"DELENS_{self.recon.lib_dir_ap}")

        if mpi.rank == 0:
            os.makedirs(self.lib_dir, exist_ok=True)

    @classmethod
    def from_ini(cls,ini):
        return cls(RecoBase.from_ini(ini))

    def get_wiener_Emode(self,idx):
        return self.recon.get_falm_sim(idx,filt='W',ret='E')

    def get_reconst_phi(self,idx):
        return self.recon.get_qlm_sim(idx) - self.recon.mean_field()

    def get_cinv_phi(self,idx,filt='W'):
        if filt == 'W':
            fname = os.path.join(self.recon.mass_dir,f"phi_wiener_sim_{idx:04d}.pkl")
        else:
            fname = os.path.join(self.recon.mass_dir,f"phi_cinv_sim_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            clpp = self.recon.cl_unl['pp'][:self.recon.Lmax+1]
            clpp = np.reshape(clpp,(1,self.recon.Lmax+1))
            Bl = np.reshape(np.ones(self.recon.Lmax+1),(1,self.recon.Lmax+1))
            phi_ = cs.utils.hp_alm2map(self.recon.nside,self.recon.Lmax,
                                          self.recon.Lmax,self.get_reconst_phi(idx))
            phi_map = np.reshape(np.array((phi_)),(1,1,self.recon.npix))
            iNL = np.reshape(cli(self.recon.norm),(1,1,self.recon.Lmax+1))
            inV = np.reshape(np.array((self.recon.mask*self.recon.extra_mask)),(1,1,self.recon.npix))
            filt_phi = cs.cninv.cnfilter_freq(1, 1, self.recon.nside, self.recon.Lmax,
                                              clpp,Bl,inV,
                                              phi_map,filter=filt,inl=iNL)
            pl.dump(filt_phi,open(fname,'wb'))
            return filt_phi
    def get_wiener_phi(self,idx):
        clpp = self.recon.cl_unl['pp'][:self.recon.Lmax+1]
        N0 = self.recon.norm
        fl = clpp/(clpp+N0)
        fl[0],fl[1] = 0,0
        #return cs.utils.almxfl(self.recon.Lmax,self.recon.Lmax,self.get_reconst_phi(idx),fl)
        return self.get_reconst_phi(idx)*fl[:self.recon.Lmax+1,None]


    def get_b_template(self,idx):
        fname = os.path.join(self.lib_dir,f"temp_sim_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            template = cs.delens.lensingb(self.recon.lmax, 10, self.recon.lmax,
                                          2, self.recon.Lmax, self.get_wiener_Emode(idx),
                                          self.get_wiener_phi(idx))
            pl.dump(template,open(fname,'wb'))
            return template
    def get_b_temp_cl(self, idx):
        return cs.utils.alm2cl(self.recon.lmax,self.get_b_template(idx))

    def get_input_Bmode(self,idx):
        qu = self.recon.get_sim(idx)
        E,B = cs.utils.hp_map2alm_spin(self.recon.nside, self.recon.lmax,
                                       self.recon.lmax, 2, qu[0][0], qu[1][0])
        del E
        return B

    def get_input_Emode(self,idx):
        qu = self.recon.get_sim(idx)
        E,B = cs.utils.hp_map2alm_spin(self.recon.nside, self.recon.lmax,
                                       self.recon.lmax, 2, qu[0][0], qu[1][0])
        del B
        return E

    def get_input_clB(self,idx):
        return cs.utils.alm2cl(self.recon.lmax,self.get_input_Bmode(idx))

    def get_TempXinput(self, idx):
        return cs.utils.alm2cl(self.recon.lmax,self.get_input_Bmode(idx),self.get_b_template(idx))

    def get_alpha(self,idx):
        return cs.utils.alm2cl(self.recon.lmax,self.get_input_Bmode(idx))/self.get_TempXinput(idx)

    def get_delensed_bmode(self,idx):
        return self.get_input_Bmode(idx) - cs.utils.almxfl(self.recon.lmax,self.recon.lmax,
                                                           self.get_b_template(idx),self.get_alpha(idx))

    def get_delensed_cl(self,idx):
        return cs.utils.alm2cl(self.recon.lmax,self.get_delensed_bmode(idx))