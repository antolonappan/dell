import curvedsky as cs
import numpy as np
import matplotlib.pyplot as plt
from filtering import Filtering
from quest import Reconstruction
import mpi
import os
import pickle as pl
import toml
from tqdm import tqdm

class Delens:

    def __init__(self,Bini,Eini,Pini,elmin,elmax,plmin,plmax):
        if (Bini == Eini) and (Eini == Pini):
            self.Pini = Reconstruction.from_ini(Pini)
            self.Bini = self.Pini.filt_lib
            self.Eini = self.Pini.filt_lib
            self.name = 'bep'
        elif (Bini == Eini) and (Bini != Pini):
            self.Bini = Filtering.from_ini(Bini)
            self.Eini = self.Bini
            self.Pini = Reconstruction.from_ini(Pini)
            self.name = 'be_p'
        elif (Eini == Pini) and (Bini != Eini):
            self.Pini = Reconstruction.from_ini(Pini)
            self.Eini = self.Pini.filt_lib
            self.Bini = Filtering.from_ini(Bini)
            self.name = 'b_ep'
        else:
            raise ValueError("DELENS ERROR: incompatible INIs")

        
        self.lib_dir = os.path.join(self.Bini.sim_lib.outfolder, f"Delensed_{self.name}")
        self.temp_dir = os.path.join(self.lib_dir, 'template')
        self.alpha_dir = os.path.join(self.lib_dir, 'alpha')
        self.delensed_dir = os.path.join(self.lib_dir, 'delensed')
        if mpi.rank == 0:
            os.makedirs(self.temp_dir, exist_ok=True)
            os.makedirs(self.alpha_dir, exist_ok=True)
            os.makedirs(self.delensed_dir, exist_ok=True)

        self.lmax = self.Bini.sim_lib.lmax
        self.elmin = elmin
        self.elmax = elmax
        self.plmin = plmin
        self.plmax = plmax
    

    @classmethod
    def from_ini(cls,ini_file):
        ini = toml.load(ini_file)
        ic = ini['ini_files']
        mc = ini['multipoles']
        return cls(ic['B'],ic['E'],ic['P'],mc['elmin'],mc['elmax'],mc['plmin'],mc['plmax'])



    def get_template(self,idx):
        fname = os.path.join(self.temp_dir, f"template_{idx}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname, 'rb'))
        else:
            Elm,_ = self.Eini.wiener_EB(idx)
            plm = self.Pini.wf_phi(idx)
            temp = cs.delens.lensingb(self.lmax,self.elmin,self.elmax,self.plmin,self.plmax,Elm,plm)
            pl.dump(temp, open(fname, 'wb'))
            return temp
            
        
