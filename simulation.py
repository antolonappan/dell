from fgbuster import harmonic_ilc_alm,CMB
import healpy as hp
import pysm3
import pysm3.units as u
import sys
import curvedsky as cs
import cmb
from database import surveys,noise
import os
import numpy as np
from tqdm import tqdm
from utils import camb_clfile,cli
import mpi
import matplotlib.pyplot as plt
import pickle as pl
from fgbuster import (Dust, Synchrotron,  # sky-fitting model
                      basic_comp_sep)
class INST:
    def __init__(self,beam,frequency):
        self.Beam = beam
        self.fwhm = beam
        self.frequency = frequency

class SimExperimentFG:

    def __init__(self,infolder,outfolder,dnside,maskpath,fwhm,fg_dir,fg_str,table,len_cl_file,Fl=0,Fh=500,which='hilc' ):

        self.infolder = infolder
        self.outfolder = outfolder
        self.lmax = (3*dnside)-1
        self.fwhm = np.radians(fwhm/60)
        self.fg_dir = fg_dir
        self.fg_str = fg_str
        table = surveys().get_table_dataframe(table)
        self.table = table[(table.frequency>Fl) & (table.frequency<Fh)]
        self.mask = hp.ud_grade(hp.read_map(maskpath,verbose=False),dnside)
        self.fsky = np.mean(self.mask)
        self.dnside = dnside
        self.Tcmb  = 2.726e6
        self.cl_len = cmb.read_camb_cls(len_cl_file,ftype='lens',output='array')[:,:self.lmax+1]
        self.which = which

        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
        print(f"using {self.infolder} and {self.fg_dir} saving to {self.outfolder}")

    def get_inv_w_noise(self):
        nP = np.around(noise(np.array(self.table.depth_p)),2)
        nT = np.around(nP/np.sqrt(2),2)
        return nT,nP

    def get_cmb(self,idx):
        fname = os.path.join(self.infolder,f"cmb_sims_{idx:04d}.fits")
        return hp.ud_grade(hp.read_map(fname,(0,1,2)),self.dnside)

    def get_fg(self,v):
        fname = os.path.join(self.fg_dir,f"{self.fg_str}_{int(v)}.fits")
        return hp.ud_grade(hp.read_map(fname,(0,1,2)),self.dnside)


    def get_noise(self,depth_i,depth_p):
        # n_pix = hp.nside2npix(self.dnside)
        # res = np.random.normal(size=(n_pix, 3))
        # depth = np.stack((depth_i, depth_p, depth_p))
        # depth *= u.arcmin * u.uK_CMB
        # depth = depth.to(
        #     getattr(u, 'uK_CMB') * u.arcmin,
        #     equivalencies=u.cmb_equivalencies(0 * u.GHz))
        # res *= depth.value / hp.nside2resol(self.dnside, True)
        # return  res.T
        t = hp.synalm(np.ones(self.lmax+1)*(np.radians(depth_i/60)**2),lmax=self.lmax)
        e = hp.synalm(np.ones(self.lmax+1)*(np.radians(depth_p/60)**2),lmax=self.lmax)
        b = hp.synalm(np.ones(self.lmax+1)*(np.radians(depth_p/60)**2),lmax=self.lmax)
        return hp.alm2map([t,e,b],nside=self.dnside)

    # def get_total_alms(self,idx,v,n_t,n_p,beam):
    #     maps = hp.smoothing(hp.smoothing(self.get_cmb(idx)+self.get_fg(v),fwhm=np.radians(beam/60.)) + self.get_noise(n_t,n_p),fwhm=self.fwhm)
    #     alms = hp.map2alm(maps*self.mask)
    #     del maps
    #     return alms
#     def get_total_alms(self,idx,v,n_t,n_p,beam,ret='alm'):
#         maps = hp.smoothing(self.get_cmb(idx)+self.get_fg(v),fwhm=np.radians(beam/60.)) + self.get_noise(n_t,n_p)
#         alms = hp.map2alm(maps*self.mask)
#         del maps
#         bl = hp.gauss_beam(np.radians(beam/60),self.lmax,pol=True).T
#         fl = hp.gauss_beam(self.fwhm,self.lmax,pol=True).T

#         hp.almxfl(alms[0],cli(bl[0]),inplace=True)
#         hp.almxfl(alms[1],cli(bl[1]),inplace=True)
#         hp.almxfl(alms[2],cli(bl[2]),inplace=True)

#         hp.almxfl(alms[0],fl[0],inplace=True)
#         hp.almxfl(alms[1],fl[1],inplace=True)
#         hp.almxfl(alms[2],fl[2],inplace=True)
#         if ret=='alm':
#             return alms
#         else:
#             return hp.alm2map(alms,self.dnside)

    def get_total_alms(self,idx,v,n_t,n_p,beam,ret='alm'):
        maps = self.get_cmb(idx)+ self.get_fg(v) + self.get_noise(n_t,n_p)
        alms = hp.map2alm(maps*self.mask)
        del maps
        bl = hp.gauss_beam(np.radians(beam/60),self.lmax)
        fl = hp.gauss_beam(self.fwhm,self.lmax)
        Bl = fl/bl
        hp.almxfl(alms[0],Bl,inplace=True)
        hp.almxfl(alms[1],Bl,inplace=True)
        hp.almxfl(alms[2],Bl,inplace=True)
        if ret=='alm':
            return alms
        else:
            return hp.alm2map(alms,self.dnside)
        
    def get_noFG_alms(self,idx):
        fsky = f"{self.fsky:.1f}".replace('.','p')
        fname = os.path.join(self.outfolder,f"exp_noFG_sims_fsky_{fsky}_{idx:04d}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname,(1,2,3))
        else:
            n_t,n_p = 1.52,2.16 #self.get_inv_w_noise()
            #maps = hp.smoothing(hp.smoothing(self.get_cmb(idx),fwhm=np.radians(.5)) + self.get_noise(n_t,n_p),fwhm=self.fwhm)
            maps = self.get_cmb(idx) + self.get_noise(n_t,n_p)
            alms = hp.map2alm(maps*self.mask)
            del maps
            bl = hp.gauss_beam(np.radians(.5),self.lmax)
            fl = hp.gauss_beam(self.fwhm,self.lmax)

            Bl = fl/bl

            hp.almxfl(alms[0],Bl,inplace=True)
            hp.almxfl(alms[1],Bl,inplace=True)
            hp.almxfl(alms[2],Bl,inplace=True)
            hp.write_alm(fname,alms)
            return alms

    def get_cinv_sim(self,idx,noFG=False):
        fsky = f"{self.fsky:.1f}".replace('.','p')
        _name = "cinv_noFG_sim" if noFG else "cinv_sim"
        fname = os.path.join(self.outfolder,f"{_name}_fsky_{fsky}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            _,sigma = self.get_inv_w_noise()
            beam = hp.gauss_beam(self.fwhm,lmax=self.lmax,pol=True).T
            Bl = np.reshape(beam[2],(1,self.lmax+1))
            invn = self.mask * (np.radians(sigma/60)/self.Tcmb)**-2
            invN = np.reshape(np.array((invn,invn)),(2,1,hp.nside2npix(self.dnside)))
            alms = self.get_noFG_alms(idx) if noFG else self.get_comp_sep_alm(idx)
            T,Q,U = hp.alm2map(alms,self.dnside)
            QU = np.reshape(np.array((Q,U)),(2,1,hp.nside2npix(self.dnside)))/self.Tcmb
            E,B = cs.cninv.cnfilter_freq(2,1,self.dnside,self.lmax,self.cl_len[1:3,:],
                                         Bl,invN,QU,chn=1,itns=[1000],eps=[1e-5],ro=10)
            pl.dump((E,B),open(fname,'wb'))
            return E, B



    def get_alms_arr(self,idx,v,n_t,n_p,beam,ret='alm'):
        
        arr = []
        for i in tqdm(range(len(v)),desc="Making alms",unit='Freq'):
            arr.append(self.get_total_alms(idx,v[i],n_t[i],n_p[i],beam[i],ret))
        return np.array(arr)

    def get_comp_sep_alm(self,idx):
        fsky = f"{self.fsky:.1f}".replace('.','p')
        fname = os.path.join(self.outfolder,f"{self.which}_sims_fsky_{fsky}_{idx:04d}.fits")
        print(fname)
        if os.path.isfile(fname):
            if self.which == 'hilc':
                return hp.read_alm(fname,(1,2,3))
            elif self.which == 'parametric':
                alms = hp.read_alm(fname,(1,2,3))
                fl = hp.gauss_beam(self.fwhm,self.lmax,pol=True).T

                hp.almxfl(alms[0],fl[0],inplace=True)
                hp.almxfl(alms[1],fl[1],inplace=True)
                hp.almxfl(alms[2],fl[2],inplace=True)
                return alms
        else:
            freqs = np.array(self.table.frequency)
            fwhm = np.array(self.table.fwhm)
            nlev_p = np.array(self.table.depth_p)
            nlev_t = nlev_p/np.sqrt(2)
            instrument = INST(None,freqs)
            if self.which == 'hilc':
                print("Harmonic ILC")
                alms = self.get_alms_arr(idx,freqs,nlev_t,nlev_p,fwhm)
                components = [CMB()]
                bins = np.arange(1000) * 10
                result = harmonic_ilc_alm(components, instrument,alms,bins)
                del alms
                alms = [result.s[0][0], result.s[0][1],result.s[0][2]]
            elif self.which == 'parametric':
                print("Parametric Method")
                maps = self.get_alms_arr(idx,freqs,nlev_t,nlev_p,fwhm,'map')
                components = [CMB(),Dust(150.), Synchrotron(20.)]
                result = basic_comp_sep(components, instrument, maps)
                alms = hp.map2alm([result.s[0][0], result.s[0][1],result.s[0][2]])
            else:
                raise NotImplementedError
            del result
            hp.write_alm(fname,alms)
            return alms

    def get_weights(self,idx):
        fname = os.path.join(self.outfolder,f"exp_weight_{idx:04d}.pkl")
        print(f"Getting Weights {idx}")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            freqs = np.array(self.table.frequency)
            fwhm = np.array(self.table.fwhm)
            nlev_p = np.array(self.table.depth_p)
            nlev_t = nlev_p/np.sqrt(2)
            alms = self.get_alms_arr(idx,freqs,nlev_t,nlev_p,fwhm)
            instrument = INST(None,freqs)
            components = [CMB()]
            bins = np.arange(1000) * 10
            result = harmonic_ilc_alm(components, instrument,alms,bins)
            w = result.W
            pk.dump(w,open(fname,'wb'))
            return w