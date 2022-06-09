import curvedsky as cs
import lenspyx
import mpi
import numpy as np
from quest import RecoBase
import os
import toml
import pickle as pl
from utils import cli
import healpy as hp



class Delens:

    def __init__(self,recon):
        self.recon = recon
        self.lib_dir = os.path.join(self.recon.lib_dir,f"DELENS_{self.recon.lib_dir_ap}")
        self.Lmax = self.recon.Lmax
        self.cl_pp = self.recon.cl_unl['pp'][:self.Lmax+1]
        self.transf = self.recon.beam
        self.nside = self.recon.nside

        if mpi.rank == 0:
            os.makedirs(self.lib_dir, exist_ok=True)

    @classmethod
    def from_ini(cls,ini):
        return cls(RecoBase.from_ini(ini))

    def get_wiener_Emode(self,idx):
        return self.recon.get_falm_sim(idx,filt='W',ret='E')

    def get_reconst_phi(self,idx):
        return self.recon.get_qlm_sim(idx) - self.recon.mean_field()

    def get_fl(self,idx):
        nhl = self.recon.norm
        fl = self.cl_pp/(self.cl_pp+ nhl)
        fl[0] = 0
        fl[1] = 0
        return fl
    def pixalm2pyalm(self,alm):
        return hp.map2alm(cs.utils.hp_alm2map(self.nside,self.Lmax,self.Lmax,alm),lmax=self.Lmax)

    def almxfl(self,alm,fl):
        return cs.utils.almxfl(self.Lmax,self.Lmax,alm,fl)

    def qlm_wf(self,idx):
        fl = self.get_fl(idx)
        return self.almxfl(self.get_reconst_phi(idx),fl)

    def deflection_field(self,idx):
        wplm = self.qlm_wf(idx)
        walpha = self.almxfl(wplm, np.sqrt(np.arange(self.Lmax + 1, dtype=float) * np.arange(1, self.Lmax + 2)))
        ftl = np.ones(self.Lmax+ 1, dtype=float) * (np.arange(self.Lmax + 1) >= 10)
        walpha = self.almxfl(walpha,ftl)
        return self.pixalm2pyalm(walpha)
    def get_template(self,idx):
        fname = os.path.join(self.lib_dir,f"temp_sim_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            elm_wf = self.pixalm2pyalm(self.get_wiener_Emode(idx))
            Q, U  = lenspyx.alm2lenmap_spin([elm_wf, None], [self.deflection_field(idx), None], self.nside, 2, facres=-1)
            del elm_wf
            Q = hp.smoothing(Q,beam_window=self.transf)*self.recon.mask
            U = hp.smoothing(U,beam_window=self.transf)*self.recon.mask
            print("Transfer function applied to the Template")
            pl.dump((Q,U),open(fname,'wb'))

            return Q, U
    def get_lensed_field(self,idx):
        return self.recon.get_sim(idx)

    def get_delensed_field(self,idx):
        fname = os.path.join(self.lib_dir,f"delens_sim_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            Q, U = self.get_template(idx)
            Q_d, U_d = self.get_lensed_field(idx)

            Q_t = Q_d - Q
            U_t = U_d - U
            pl.dump((Q_t,U_t),open(fname,'wb'))
            del(Q,U,Q_d,U_d)

            return Q_t, U_t