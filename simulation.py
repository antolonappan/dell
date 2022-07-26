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
import toml


class INST:
    def __init__(self,beam,frequency):
        self.Beam = beam
        self.fwhm = beam
        self.frequency = frequency

class SimExperimentFG:

    def __init__(self,infolder,outfolder,dnside,fg_dir,fg_str,table,len_cl_file,nsim,Fl=0,Fh=500,noFG=False):

        self.infolder = infolder
        outfolder = os.path.join(outfolder,f"{fg_str}") if not noFG else os.path.join(outfolder,f"noFG")
        self.outfolder = outfolder
        self.mapfolder = os.path.join(outfolder, 'Maps')
        self.noisefolder = os.path.join(outfolder, 'Noise')
        self.weightfolder = os.path.join(outfolder, 'Weights')
        self.lmax = (3*dnside)-1
        self.fg_dir = fg_dir
        self.fg_str = fg_str
        table = surveys().get_table_dataframe(table)
        self.table = table[(table.frequency>Fl) & (table.frequency<Fh)]
        self.dnside = dnside
        self.Tcmb  = 2.726e6
        self.cl_len = cmb.read_camb_cls(len_cl_file,ftype='lens',output='array')[:,:self.lmax+1]
        self.noFG = noFG
        self.nsim = nsim

        if mpi.rank == 0:
            os.makedirs(self.mapfolder,exist_ok=True)
            os.makedirs(self.noisefolder,exist_ok=True)
            os.makedirs(self.weightfolder,exist_ok=True)
        print(f"using {self.infolder} and {self.fg_dir} saving to {outfolder}")

     

    @classmethod
    def from_ini(cls,ini_file):
        config = toml.load(ini_file)
        mc = config['Map']
        infolder = mc['infolder']
        print(infolder)
        outfolder = mc['outfolder']
        dnside = mc['nside']
        fg_dir = mc['fg_dir']
        fg_str = mc['fg_str']
        table = mc['table']
        len_cl_file = mc['len_cl_file']
        Fl = float(mc['Fl'])
        Fh = float(mc['Fh'])
        noFG = bool(mc['noFG'])
        nsim = int(mc['nsim'])
        return cls(infolder,outfolder,dnside,fg_dir,fg_str,table,len_cl_file,nsim,Fl,Fh,noFG)


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

    def convolved_cmb_fg_vect(self,idx):
        fname = os.path.join(self.mapfolder,f"tmp_cmb_{idx:04d}.pkl")
        if not os.path.exists(fname):
            V = np.array(self.table.frequency)
            Beam = np.array(self.table.fwhm)
            maps = []
            for v,b in tqdm(zip(V,Beam),desc="Making maps",unit='Freq'):
                maps.append(hp.smoothing(self.get_cmb(idx) + self.get_fg(v), fwhm=np.radians(b/60)))
            pl.dump(maps,open(fname,'wb'))
        else:
            maps = pl.load(open(fname,'rb'))
        return np.array(maps)
    
    def convolved_cmb_vect(self,idx):
        fname = os.path.join(self.mapfolder,f"tmp_cmb_{idx:04d}.pkl")
        if not os.path.exists(fname):
            Beam = np.array(self.table.fwhm)
            maps = []
            for b in tqdm(Beam,desc="Making maps",unit='Freq'):
                maps.append(hp.smoothing(self.get_cmb(idx), fwhm=np.radians(b/60)))
            pl.dump(maps,open(fname,'wb'))
        else:
            maps = pl.load(open(fname,'rb'))
        return np.array(maps)

    def get_noise_map(self,idx):
        fname = os.path.join(self.noisefolder,f"tmp_noise_{idx:04d}.pkl")
        if not os.path.exists(fname):
            depth_p =np.array(self.table.depth_p)
            depth_i = depth_p/np.sqrt(2)
            pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.dnside)) * (180. * 60. / np.pi) ** 2
            """sigma_pix_I/P is std of noise per pixel. It is an array of length
            equal to the number of input maps."""
            sigma_pix_I = np.sqrt(depth_i ** 2 / pix_amin2)
            sigma_pix_P = np.sqrt(depth_p ** 2 / pix_amin2)
            npix = hp.nside2npix(self.dnside)
            noise = np.random.randn(len(depth_i), 3, npix)
            noise[:, 0, :] *= sigma_pix_I[:, None]
            noise[:, 1, :] *= sigma_pix_P[:, None]
            noise[:, 2, :] *= sigma_pix_P[:, None]
            pl.dump(noise, open(fname, 'wb'))
        else:
            noise = pl.load(open(fname, 'rb'))
        
        return noise

    def get_cmb_alms(self,idx,ret=0):
        mapfile = os.path.join(self.mapfolder,f"sims_{idx:04d}.fits")
        noisefile = os.path.join(self.noisefolder,f"sims_{idx:04d}.fits")
        weightfile = os.path.join(self.weightfolder,f"sims_{idx:04d}.pkl")

        if (not os.path.isfile(mapfile)) or  \
           (not os.path.isfile(noisefile)) or \
           (not os.path.isfile(weightfile)):
            instrument = INST(None,np.array(self.table.frequency))
            components = [CMB()]
            bins = np.arange(1000) * 10
            Beam = np.array(self.table.fwhm)

            noise = self.get_noise_map(idx)
            if self.noFG:
                print("No foregrounds")
                maps = self.convolved_cmb_vect(idx) + noise
            else:
                print("Using foregrounds")
                maps = self.convolved_cmb_fg_vect(idx) + noise

            map_alms = []
            noise_alms = []
            for m,n,b in tqdm(zip(maps,noise,Beam),desc="Making alms",unit='Freq'):
                alms_ = hp.map2alm(m)
                fl = hp.gauss_beam(np.radians(b/60),lmax=self.lmax,pol=True).T
                hp.almxfl(alms_[0],cli(fl[0]),inplace=True)
                hp.almxfl(alms_[1],cli(fl[1]),inplace=True)
                hp.almxfl(alms_[2],cli(fl[2]),inplace=True)
                map_alms.append(alms_)
                noise_alms.append(hp.map2alm(n))
            del (maps,noise)

            result = harmonic_ilc_alm(components, instrument,np.array(map_alms),bins)

            del map_alms

            weights = result.W

            pl.dump(weights,open(weightfile,'wb'))

            cmb_final = np.array([result.s[0,0],result.s[0,1],result.s[0,2]])
            noise_final = self.apply_harmonic_W(result.W,np.array(noise_alms))

            del noise_alms
            hp.write_alm(mapfile,cmb_final)
            hp.write_alm(noisefile,noise_final[0])
            print("removing tmp files")
            os.remove(os.path.join(self.noisefolder,f"tmp_noise_{idx:04d}.pkl"))
            os.remove(os.path.join(self.mapfolder,f"tmp_cmb_{idx:04d}.pkl"))

            return cmb_final,noise_final,weights

        else:
            cmb_final =  hp.read_alm(mapfile,(1,2,3))
            noise_final = hp.read_alm(noisefile,(1,2,3))
            weights = pl.load(open(weightfile,"rb"))
            return cmb_final,noise_final,weights

    
        

    def apply_harmonic_W(self,W, alms):  
        lmax = hp.Alm.getlmax(alms.shape[-1])
        res = np.full((W.shape[-2],) + alms.shape[1:], np.nan, dtype=alms.dtype)
        start = 0
        for i in range(0, lmax+1):
            n_m = lmax + 1 - i
            res[..., start:start+n_m] = np.einsum('...lcf,f...l->c...l',
                                                W[..., i:, :, :],
                                                alms[..., start:start+n_m])
            start += n_m
        return res

    def run_job(self):
        jobs = np.arange(mpi.size)
        for i in jobs[mpi.rank::mpi.size]:
            Null = self.get_cmb_alms(i)

    def get_cleaned_cmb(self,idx):
        alms = hp.read_alm(os.path.join(self.mapfolder,f"sims_{idx:04d}.fits"),(1,2,3))
        return alms
    
    def get_noise_cmb(self,idx):
        alms = hp.read_alm(os.path.join(self.noisefolder,f"sims_{idx:04d}.fits"),(1,2,3))
        return alms

    def get_weights_cmb(self,idx):
        weights = pl.load(open(os.path.join(self.weightfolder,f"sims_{idx:04d}.pkl"),"rb"))
        return weights
    
    def plot_cl(self,idx):
        cmb = self.get_cleaned_cmb(idx)
        noise = self.get_noise_cmb(idx)
        clee = hp.alm2cl(cmb[1])
        clbb = hp.alm2cl(cmb[2])
        clbb_noise = hp.alm2cl(noise[2])
        clee_noise = hp.alm2cl(noise[1])
        plt.figure(figsize=(8,8))
        plt.loglog(clbb,label="BB")
        plt.loglog(clbb_noise,label="BB noise")
        plt.loglog(clee,label="EE")
        plt.loglog(clee_noise,label="EE noise")
        plt.loglog(self.cl_len[2,:]*self.Tcmb**2 ,label="BB CAMB")
        plt.loglog(self.cl_len[1,:]*self.Tcmb**2 ,label="EE CAMB")
        plt.legend()

    def noise_mean_mpi(self):
        fname = os.path.join(self.outfolder,f"noise_mean_{mpi.size}.pkl")
        if not os.path.isfile(fname):
            noise_alm = self.get_noise_cmb(mpi.rank)
            nlt = hp.alm2cl(noise_alm[0])
            nle = hp.alm2cl(noise_alm[1])
            nlb = hp.alm2cl(noise_alm[2])
            
            if mpi.rank == 0:
                total_nlt = np.zeros(nlt.shape)
                total_nle = np.zeros(nle.shape)
                total_nlb = np.zeros(nlb.shape)
            else:
                total_nlt = None
                total_nle = None
                total_nlb = None
            
            mpi.com.Reduce(nlt,total_nlt, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(nle,total_nle, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(nlb,total_nlb, op=mpi.mpi.SUM,root=0)

            if mpi.rank == 0:
                mean_nlt = total_nlt/mpi.size
                mean_nle = total_nle/mpi.size
                mean_nlb = total_nlb/mpi.size
                pl.dump([mean_nlt,mean_nle,mean_nlb],open(fname,'wb'))

            mpi.barrier()
    
    def noise_spectra(self,num):
        fname = os.path.join(self.outfolder,f"noise_mean_{num}.pkl")
        return  pl.load(open(fname,'rb'))
        
    
    def plot_noise_spectra(self,num):
        tl,el,bl = self.noise_spectra(num)
        plt.figure(figsize=(8,8))
        plt.loglog(tl,label="T")
        plt.loglog(el,label="E")
        plt.loglog(bl,label="B")
        plt.loglog(self.cl_len[0,:]*self.Tcmb**2 ,label="T")
        plt.loglog(self.cl_len[1,:]*self.Tcmb**2 ,label="E")
        plt.loglog(self.cl_len[2,:]*self.Tcmb**2 ,label="B")
        plt.axhline(np.radians(2.16/60)**2,color="k",ls="--")
        plt.legend()

    def get_beam_sim(self,idx):
        W = self.get_weights_cmb(idx)
        Wt = W[0]
        We = W[1]
        Wb = W[2]

        Beam = []
        for i in np.array(self.table.fwhm):
            Beam.append(hp.gauss_beam(np.radians(i/60), lmax=self.lmax))
        Beam = np.array(Beam)

        bl_tt,bl_ee,bl_bb = np.zeros(self.lmax+1),np.zeros(self.lmax+1),np.zeros(self.lmax+1)
        for i in range(self.lmax+1):
            bl_tt[i] = np.dot(Beam[:, i], Wt[i][0])
            bl_ee[i] = np.dot(Beam[:, i], We[i][0])
            bl_bb[i] = np.dot(Beam[:, i], Wb[i][0])
        bl_tt = np.array(bl_tt)
        bl_ee = np.array(bl_ee)
        bl_bb = np.array(bl_bb)
        return bl_tt,bl_ee,bl_bb

    def beam_mean_mpi(self):
        fname = os.path.join(self.outfolder,f"beam_mean_{mpi.size}.pkl")
        if not os.path.isfile(fname):
            bt,be,bb = self.get_beam_sim(mpi.rank)
            if mpi.rank == 0:
                total_bt = np.zeros(bt.shape)
                total_be = np.zeros(be.shape)
                total_bb = np.zeros(bb.shape)
            else:
                total_bt = None
                total_be = None
                total_bb = None
            
            mpi.com.Reduce(bt,total_bt, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(be,total_be, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(bb,total_bb, op=mpi.mpi.SUM,root=0)

            if mpi.rank == 0:
                mean_bt = total_bt/mpi.size
                mean_be = total_be/mpi.size
                mean_bb = total_bb/mpi.size
                pl.dump([mean_bt,mean_be,mean_bb],open(fname,'wb'))

            mpi.barrier()

    def beam_spectra(self,num):
        fname = os.path.join(self.outfolder,f"beam_mean_{num}.pkl")
        return  pl.load(open(fname,'rb'))

    


        



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-maps', dest='maps', action='store_true', help='map making')
    parser.add_argument('-noise', dest='noise', action='store_true', help='noise making')
    parser.add_argument('-beam', dest='beam', action='store_true', help='noise making')

    args = parser.parse_args()
    ini = args.inifile[0]

    sim = SimExperimentFG.from_ini(ini)

    if args.maps:
        sim.run_job()

    if args.noise:
        sim.noise_mean_mpi()

    if args.beam:
        sim.beam_mean_mpi()

    mpi.barrier()


