from fgbuster import harmonic_ilc_alm,CMB
import healpy as hp
import cmb
from database import Surveys
from utils import noise,ini_full
import os
import numpy as np
from tqdm import tqdm
from utils import camb_clfile,cli
import mpi
import matplotlib.pyplot as plt
import pickle as pl
import lenspyx
import toml
import pysm3
import pysm3.units as u




class INST:
    def __init__(self,beam,frequency):
        self.Beam = beam
        self.fwhm = beam
        self.frequency = frequency

class SimExperimentFG:
    """
    Simulation Library
    ******************
    infolder : str : path to the folder containing the CMB only simulation .fits files
    outfolder: str : path to the folder where the component separated simulation will be saved
    dnside: int : resolution of the maps
    fg_dir: str : path to the folder containing the foreground simulation .fits files
    fg_str: str : string to identify the foreground simulation
    table : str : table containing the survey information
    len_cl_file: str : path to the lensing cl file
    nsim: int : number of simulations to be generated
    Fl: float : lower frequency limit
    Fh: float : higher frequency limit
    noFG: bool : if True, no foreground is used
    """

    def __init__(self,infolder,outfolder,dnside,fg_dir,fg_str,table,len_cl_file,nsim,Fl=0,Fh=500,noFG=False,verbose=False):

        self.infolder = infolder
        outfolder = os.path.join(outfolder,f"{fg_str}") if not noFG else os.path.join(outfolder,f"noFG")
        self.outfolder = outfolder
        self.mapfolder = os.path.join(outfolder, 'Maps')
        self.noisefolder = os.path.join(outfolder, 'Noise')
        self.weightfolder = os.path.join(outfolder, 'Weights')
        self.lmax = (3*dnside)-1
        self.fg_dir = fg_dir
        self.fg_str = fg_str
        table = Surveys().get_table_dataframe(table)
        self.table = table[(table.frequency>Fl) & (table.frequency<Fh)]
        self.dnside = dnside
        self.Tcmb  = 2.726e6
        self.cl_len = cmb.read_camb_cls(len_cl_file,ftype='lens',output='array')[:,:self.lmax+1]  # type: ignore
        self.noFG = noFG
        self.nsim = nsim

        if mpi.rank == 0:
            os.makedirs(self.mapfolder,exist_ok=True)
            os.makedirs(self.noisefolder,exist_ok=True)
            os.makedirs(self.weightfolder,exist_ok=True)

        self.verbose = verbose

        self.vprint(f"SIMULATION INFO: CMB Realisation - {self.infolder}")
        self.vprint(f"SIMULATION INFO: Foreground - {self.fg_dir}")
        self.vprint(f"SIMULATION INFO: Foreground Model - {self.fg_str}")
        self.vprint(f"SIMULATION INFO: Foreground included - {not self.noFG}")
        self.vprint(f"SIMULATION INFO: Number of simulations - {self.nsim}")
        self.vprint(f"SIMULATION INFO: Frequency range - {Fl} GHz - {Fh} GHz")
        self.vprint(f"SIMULATION INFO: NSIDE - {self.dnside}")
        self.vprint(f"SIMULATION INFO: Output folder - {self.outfolder}")
        print(f"SIMUALATION object with {'out' if self.noFG else ''} FG: Loaded")

    def vprint(self,txt):
        if self.verbose:
            print(txt)

    @classmethod
    def from_ini(cls,ini_file,verbose=False):
        """
        Initialize the class from an ini file
        """
        config = toml.load(ini_full(ini_file))
        mc = config['Map']
        infolder = mc['infolder']
        outfolder = mc['outfolder']
        dnside = mc['nside']
        fg_dir = mc['fg_dir']
        fg_str = mc['fg_str']
        table = mc['table']
        len_cl_file = mc['len_cl_file']
        Fl = int(mc['Fl'])
        Fh = int(mc['Fh'])
        noFG = bool(mc['noFG'])
        nsim = int(mc['nsim'])
        return cls(infolder,outfolder,dnside,fg_dir,fg_str,table,len_cl_file,nsim,Fl,Fh,noFG,verbose)


    def get_inv_w_noise(self):
        """
        Get the inverse weighted noise level
        """
        nP = np.around(noise(np.array(self.table.depth_p)),2)
        nT = np.around(nP/np.sqrt(2),2)
        return nT,nP

    def get_cmb(self,idx):
        """
        Get the CMB map of the given simulation
        """
        fname = os.path.join(self.infolder,f"cmb_sims_{idx:04d}.fits")
        alms = hp.map2alm(hp.read_map(fname,(0,1,2)),lmax=self.lmax)
        maps = hp.alm2map(alms,self.dnside)
        del alms
        return maps

    def get_fg(self,v):
        """
        Get the foreground map
        """
        fname = os.path.join(self.fg_dir,f"{self.fg_str}_{int(v)}.fits")
        return hp.ud_grade(hp.read_map(fname,(0,1,2)),self.dnside) # type: ignore

    def convolved_cmb_fg_vect(self,idx):
        """
        frequency maps of  cmb + foreground convolved with the beam
        """
        fname = os.path.join(self.mapfolder,f"tmp_cmb_{idx:04d}.pkl")
        if not os.path.exists(fname):
            V = np.array(self.table.frequency)
            Beam = np.array(self.table.fwhm)
            maps = []
            cmb = self.get_cmb(idx)
            for v,b in tqdm(zip(V,Beam),desc="Making maps",unit='Freq',leave=True):
                maps.append(hp.smoothing(cmb + self.get_fg(v), fwhm=np.radians(b/60)))
            pl.dump(maps,open(fname,'wb'))
            del cmb
        else:
            maps = pl.load(open(fname,'rb'))
        return np.array(maps)
    
    def convolved_cmb_vect(self,idx):
        """
        frequency maps of  cmb convolved with the beam

        """
        fname = os.path.join(self.mapfolder,f"tmp_cmb_{idx:04d}.pkl")
        if not os.path.exists(fname):
            Beam = np.array(self.table.fwhm)
            maps = []
            cmb = self.get_cmb(idx)
            for b in tqdm(Beam,desc="Making maps",unit='Freq'):
                maps.append(hp.smoothing(cmb, fwhm=np.radians(b/60)))
            pl.dump(maps,open(fname,'wb'))
            del cmb
        else:
            maps = pl.load(open(fname,'rb'))
        return np.array(maps)

    def get_noise_map(self,idx):
        """
        Noise map of all frequency channels of the given simulation
        """
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

    def get_cmb_alms(self,idx,ret=False,binwidth=10):
        """
        Harmonic ILC component seperated CMB alms
        """
        mapfile = os.path.join(self.mapfolder,f"sims_{idx:04d}.fits")
        noisefile = os.path.join(self.noisefolder,f"sims_{idx:04d}.fits")
        weightfile = os.path.join(self.weightfolder,f"sims_{idx:04d}.pkl")

        if (not os.path.isfile(mapfile)) or  \
           (not os.path.isfile(noisefile)) or \
           (not os.path.isfile(weightfile)):
            instrument = INST(None,np.array(self.table.frequency))
            components = [CMB()]
            bins = np.arange(1000) * binwidth
            Beam = np.array(self.table.fwhm)

            noise = self.get_noise_map(idx)
            if self.noFG:
                maps = self.convolved_cmb_vect(idx) + noise
            else:
                maps = self.convolved_cmb_fg_vect(idx) + noise

            map_alms = []
            noise_alms = []
            for i in tqdm(range(len(Beam)),desc="Making alms",unit='Freq',leave=True):
                alms_ = hp.map2alm(maps[i])
                n_ = hp.map2alm(noise[i])
                fl = hp.gauss_beam(np.radians(Beam[i]/60),lmax=self.lmax,pol=True).T
                hp.almxfl(alms_[0],cli(fl[0]),inplace=True)
                hp.almxfl(alms_[1],cli(fl[1]),inplace=True)
                hp.almxfl(alms_[2],cli(fl[2]),inplace=True)
                hp.almxfl(n_[0],cli(fl[0]),inplace=True)
                hp.almxfl(n_[1],cli(fl[1]),inplace=True)
                hp.almxfl(n_[2],cli(fl[2]),inplace=True)
                map_alms.append(alms_)
                noise_alms.append(n_)
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
            self.vprint("SIMULATION INFO: removing tmp files")
            os.remove(os.path.join(self.noisefolder,f"tmp_noise_{idx:04d}.pkl"))
            os.remove(os.path.join(self.mapfolder,f"tmp_cmb_{idx:04d}.pkl"))
            
            if ret:
                return cmb_final,noise_final,weights
            else:
                del (cmb_final,noise_final,weights)
                return 0

        else:
            if ret:
                cmb_final =  hp.read_alm(mapfile,(1,2,3)) # type: ignore
                noise_final = hp.read_alm(noisefile,(1,2,3)) # type: ignore
                weights = pl.load(open(weightfile,"rb"))
                return cmb_final,noise_final,weights
            else:
                return 0

    
        

    def apply_harmonic_W(self,W, alms): 
        """
        Helper function to apply the harmonic weights to the alms
        """ 
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

    def run_job_mpi(self):
        """
        MPI Job to run the component seperation
        """
        jobs = np.arange(self.nsim)
        for i in jobs[mpi.rank::mpi.size]:
            Null = self.get_cmb_alms(i)
            del Null
        mpi.barrier()

    def run_job(self):
        job = np.arange(self.nsim)
        for i in tqdm(job,desc='Component Separation',unit='sim'):
            Null = self.get_cmb_alms(i)
            del Null

    def get_cleaned_cmb(self,idx):
        """
        To read the cleaned cmb map
        """

        alms = hp.read_alm(os.path.join(self.mapfolder,f"sims_{idx:04d}.fits"),(1,2,3)) # type: ignore
        return alms
    
    def get_noise_cmb(self,idx):
        """
        To read the noise cmb map
        """
        alms = hp.read_alm(os.path.join(self.noisefolder,f"sims_{idx:04d}.fits"),(1,2,3)) # type: ignore
        return alms

    def get_weights_cmb(self,idx):
        """
        To read the weights cmb map
        """
        weights = pl.load(open(os.path.join(self.weightfolder,f"sims_{idx:04d}.pkl"),"rb"))
        return weights
    
    def plot_cl(self,idx):
        cmb = self.get_cleaned_cmb(idx)
        noise = self.get_noise_cmb(idx)
        clee = hp.alm2cl(cmb[1])
        clbb = hp.alm2cl(cmb[2]) # type: ignore
        clbb_noise = hp.alm2cl(noise[2]) # type: ignore
        clee_noise = hp.alm2cl(noise[1])
        plt.figure(figsize=(8,8))
        plt.loglog(clbb,label="BB")
        plt.loglog(clbb_noise,label="BB noise")
        plt.loglog(clee,label="EE")
        plt.loglog(clee_noise,label="EE noise")
        plt.loglog(self.cl_len[2,:]*self.Tcmb**2 ,label="BB CAMB")
        plt.loglog(self.cl_len[1,:]*self.Tcmb**2 ,label="EE CAMB")
        plt.legend()
    
    def cl_stat(self,n=100):
        """
        To calculate the mean and std of the cl
        """
        fname = os.path.join(self.outfolder,f"cl_stat_{n}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,"rb"))
        else:
            clbb = np.zeros((n,self.lmax+1))
            clee = np.zeros((n,self.lmax+1))
            for i in tqdm(range(n),desc="Calculating Cl",unit='sim'):
                cmb = self.get_cleaned_cmb(i)
                clee[i,:] = hp.alm2cl(cmb[1])
                clbb[i,:] = hp.alm2cl(cmb[2]) # type: ignore
            stat = (np.mean(clee,axis=0),np.std(clee,axis=0),np.mean(clbb,axis=0),np.std(clbb,axis=0))
            pl.dump(stat,open(fname,"wb"))
            return stat

    def plot_stat(self,n=100):
        """
        To plot the mean and std of the cl
        """
        stat = self.cl_stat(n)
        plt.figure(figsize=(8,8))
        plt.loglog(stat[0],label="$\\frac{1}{b_\ell^2}\\left(C_\ell^{EE} + C_\ell^{FG\;res} + N_\ell\\right)$",c='r',ls='--',lw=2) # type: ignore
        plt.fill_between(np.arange(self.lmax+1),stat[0]-(stat[1]),stat[0]+stat[1],color='r',alpha=0.5)
        plt.loglog(stat[2],label="$\\frac{1}{b_\ell^2}\\left(C_\ell^{BB} + C_\ell^{FG\;res} + N_\ell\\right)$",c='b',ls='--',lw=2) # type: ignore
        plt.fill_between(np.arange(self.lmax+1),stat[2]-stat[3],stat[2]+stat[3],color='b',alpha=0.5)
        plt.loglog(self.cl_len[1,:]*self.Tcmb**2 ,c='k',ls='--',lw=2,label="EE")
        plt.loglog(self.cl_len[2,:]*self.Tcmb**2 ,c='k',ls='-.',lw=2,label="BB")
        plt.xlim(2,800)
        plt.ylim(1e-7,10)
        plt.legend(ncol=2,fontsize=20)
        plt.xlabel(r"$\ell$",fontsize=20)
        plt.ylabel(r"$C_\ell$ [$\mu K^2$]",fontsize=20)
        plt.tick_params(labelsize=20)
        plt.savefig('cl_stat.pdf',bbox_inches='tight',dpi=300)


    def noise_mean_mpi(self):
        """
        MPI Job to calculate the noise mean
        """
        if mpi.size>1:
            fname = os.path.join(self.outfolder,f"noise_mean_{mpi.size}.pkl")
        else:
            fname = os.path.join(self.outfolder,f"noise_mean_{self.nsim}.pkl")
        if not os.path.isfile(fname):
            noise_alm = self.get_noise_cmb(mpi.rank)
            nlt = hp.alm2cl(noise_alm[0])
            nle = hp.alm2cl(noise_alm[1])
            nlb = hp.alm2cl(noise_alm[2]) # type: ignore
            
            if mpi.rank == 0:
                total_nlt = np.zeros(nlt.shape)
                total_nle = np.zeros(nle.shape)
                total_nlb = np.zeros(nlb.shape)
            else:
                total_nlt = None
                total_nle = None
                total_nlb = None
            
            if mpi.size > 1:
                mpi.com.Reduce(nlt,total_nlt, op=mpi.mpi.SUM,root=0)
                mpi.com.Reduce(nle,total_nle, op=mpi.mpi.SUM,root=0)
                mpi.com.Reduce(nlb,total_nlb, op=mpi.mpi.SUM,root=0)
                mpi.barrier()
                if mpi.rank == 0:
                    mean_nlt = total_nlt/mpi.size # type: ignore
                    mean_nle = total_nle/mpi.size # type: ignore
                    mean_nlb = total_nlb/mpi.size # type: ignore
            else:
                for i in tqdm(range(self.nsim),desc='Computing Noise Spectra mean',unit='sim'):
                    noise_alm = self.get_noise_cmb(mpi.rank)
                    total_nlt += hp.alm2cl(noise_alm[0])
                    total_nle += hp.alm2cl(noise_alm[1])
                    total_nlb += hp.alm2cl(noise_alm[2])
                    del noise_alm

                mean_nlt = total_nlt/self.nsim # type: ignore
                mean_nle = total_nle/self.nsim# type: ignore
                mean_nlb = total_nlb/self.nsim


            pl.dump([mean_nlt,mean_nle,mean_nlb],open(fname,'wb'))



    def noise_spectra(self,num):
        """
        To read the noise mean
        """
        fname = os.path.join(self.outfolder,f"noise_mean_{num}.pkl")
        return  np.array(pl.load(open(fname,'rb')))/self.Tcmb**2
        
    
    def plot_noise_spectra(self,num):
        tl,el,bl = self.noise_spectra(num)
        plt.figure(figsize=(8,8))
        plt.loglog(tl*self.Tcmb**2,label="T")
        plt.loglog(el*self.Tcmb**2,label="E")
        plt.loglog(bl*self.Tcmb**2,label="B")
        plt.loglog(self.cl_len[0,:]*self.Tcmb**2 ,label="T")
        plt.loglog(self.cl_len[1,:]*self.Tcmb**2 ,label="E")
        plt.loglog(self.cl_len[2,:]*self.Tcmb**2 ,label="B")
        plt.axhline(np.radians(2.16/60)**2,color="k",ls="--")
        plt.legend()

    def get_beam_sim(self,idx):
        """
        To get the Effective beam for the simulation using ILC weights
        """
        self.vprint("Getting beam for simulation")
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
        """
        MPI Job to calculate the effective beam mean
        """
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
                mean_bt = total_bt/mpi.size # type: ignore
                mean_be = total_be/mpi.size # type: ignore
                mean_bb = total_bb/mpi.size # type: ignore
                pl.dump([mean_bt,mean_be,mean_bb],open(fname,'wb'))

            mpi.barrier()

    def beam_spectra(self,num):
        """
        To read the effective beam mean
        """
        fname = os.path.join(self.outfolder,f"beam_mean_{num}.pkl")
        return  pl.load(open(fname,'rb'))
    
    def fg_vect(self):
        """
        To get all foreground alms
        """
        fg_alm = []
        for v in tqdm(np.array(self.table.frequency),desc="Getting Foreground",unit="freq",):
            fg_alm.append(hp.map2alm(self.get_fg(v)))
        
        return np.array(fg_alm)

    def fg_res(self,idx):
        """
        To get the foreground residuals
        """
        fg_alm = self.fg_vect()
        fg_res = self.apply_harmonic_W(self.get_weights_cmb(idx),fg_alm)
        del fg_alm
        return fg_res

    def fg_res_mean_mpi(self):
        """
        MPI Job to calculate the foreground residual mean
        """
        fname = os.path.join(self.outfolder,f"fg_res_mean_{mpi.size}.pkl")
        if not os.path.isfile(fname):
            fg_res = self.fg_res(mpi.rank)[0]
            ft = hp.alm2cl(fg_res[0])
            fe = hp.alm2cl(fg_res[1])
            fb = hp.alm2cl(fg_res[2])
            del fg_res
            if mpi.rank == 0:
                total_ft = np.zeros(ft.shape)
                total_fe = np.zeros(fe.shape)
                total_fb = np.zeros(fb.shape)
            else:
                total_ft = None
                total_fe = None
                total_fb = None
            
            mpi.com.Reduce(ft,total_ft, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(fe,total_fe, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(fb,total_fb, op=mpi.mpi.SUM,root=0)

            if mpi.rank == 0:
                mean_ft = total_ft/mpi.size # type: ignore
                mean_fe = total_fe/mpi.size # type: ignore
                mean_fb = total_fb/mpi.size # type: ignore
                pl.dump([mean_ft,mean_fe,mean_fb],open(fname,'wb'))

            mpi.barrier()
    
    def fg_res_mean(self,num):
        """
        To read the foreground residual mean
        """
        fname = os.path.join(self.outfolder,f"fg_res_mean_{num}.pkl")
        return  np.array(pl.load(open(fname,'rb')))/self.Tcmb**2

    def cmb_mean_mpi(self):
        """
        MPI Job to calculate the noise mean
        """
        fname = os.path.join(self.outfolder,f"cmb_mean_{mpi.size}.pkl")
        if not os.path.isfile(fname):
            cmb_alm = self.get_cleaned_cmb(mpi.rank)
            tt = hp.alm2cl(cmb_alm[0])
            ee = hp.alm2cl(cmb_alm[1])
            bb = hp.alm2cl(cmb_alm[2]) # type: ignore
            
            if mpi.rank == 0:
                total_tt = np.zeros(tt.shape)
                total_ee = np.zeros(ee.shape)
                total_bb = np.zeros(bb.shape)
            else:
                total_tt = None
                total_ee = None
                total_bb = None
            
            mpi.com.Reduce(tt,total_tt, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(ee,total_ee, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(bb,total_bb, op=mpi.mpi.SUM,root=0)

            if mpi.rank == 0:
                mean_tt = total_tt/mpi.size # type: ignore
                mean_ee = total_ee/mpi.size # type: ignore
                mean_bb = total_bb/mpi.size # type: ignore
                pl.dump([mean_tt,mean_ee,mean_bb],open(fname,'wb'))

            mpi.barrier()
    
    def cmb_mean(self,num):
        """
        To read the noise mean
        """
        fname = os.path.join(self.outfolder,f"cmb_mean_{num}.pkl")
        return  np.array(pl.load(open(fname,'rb')))/self.Tcmb**2

class CMBLensed:
    """
    Lensing class:
    It saves seeds, Phi Map and Lensed CMB maps
    
    """
    def __init__(self,outfolder,nsim,cl_path,scal_file,pot_file,phi='v',verbose=False):
        self.outfolder = outfolder
        self.cl_unl = camb_clfile(os.path.join(cl_path, scal_file))
        self.cl_pot = camb_clfile(os.path.join(cl_path, pot_file))
        self.nside = 2048
        self.lmax = 4096
        self.dlmax = 1024
        self.facres = 0
        self.verbose = verbose
        self.nsim = nsim
        self.phi = phi        
        #folder for CMB
        self.cmb_dir = os.path.join(self.outfolder,f"CMB")
        #folder for mass
        self.mass_dir = os.path.join(self.outfolder,f"MASS") 
        
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            os.makedirs(self.mass_dir,exist_ok=True) 
            os.makedirs(self.cmb_dir,exist_ok=True)
        
        
        fname = os.path.join(self.outfolder,'seeds.pkl')
        if (not os.path.isfile(fname)) and (mpi.rank == 0):
            seeds = self.get_seeds
            pl.dump(seeds, open(fname,'wb'), protocol=2)
        mpi.barrier()
        self.seeds = pl.load(open(fname,'rb'))

        self.vprint(f"simulations are made with {self.phi} phi")
        # if self.phi == 'c':
        #     NULL = self.get_phi(0)
        #     del NULL
        
        
    @property
    def get_seeds(self):
        """
        non-repeating seeds
        """
        seeds =[]
        no = 0
        while no <= self.nsim-1:
            r = np.random.randint(11111,99999)
            if r not in seeds:
                seeds.append(r)
                no+=1
        return seeds
    
    def vprint(self,string):
        if self.verbose:
            print(string)
                  
    def get_phi(self,idx):
        """
        set a seed
        generate phi_LM
        Save the phi
        """
        if self.phi == 'v':
            fname = os.path.join(self.mass_dir,f"phi_sims_{idx:04d}.fits")
        elif self.phi == 'c':
            fname = os.path.join(self.mass_dir,f"phi_sims.fits")
        else:
            raise ValueError(f"{self.phi} is not a valid phi")
    
        if os.path.isfile(fname):
            self.vprint(f"Phi field from cache: {idx}")
            return hp.read_alm(fname)
        else:
            np.random.seed(self.seeds[idx])
            plm = hp.synalm(self.cl_pot['pp'], lmax=self.lmax + self.dlmax, new=True)
            hp.write_alm(fname,plm)
            self.vprint(f"Phi field cached: {idx}")
            return plm
        
    def get_kappa(self,idx):
        """
        generate deflection field
        sqrt(L(L+1)) * phi_{LM}
        """
        der = np.sqrt(np.arange(self.lmax + 1, dtype=float) * np.arange(1, self.lmax + 2))
        return hp.almxfl(self.get_phi(idx), der)
    
    def get_unlensed_alm(self,idx):
        self.vprint(f"Synalm-ing the Unlensed CMB temp: {idx}")
        Cls = [self.cl_unl['tt'],self.cl_unl['ee'],self.cl_unl['tt']*0,self.cl_unl['te']]
        np.random.seed(self.seeds[idx]+1)
        alms = hp.synalm(Cls,lmax=self.lmax + self.dlmax,new=True)
        return alms
    
    def get_lensed(self,idx):
        fname = os.path.join(self.cmb_dir,f"cmb_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"CMB fields from cache: {idx}")
            return hp.read_map(fname,(0,1,2),dtype=np.float64) # type: ignore 
        else:
            dlm = self.get_kappa(idx)
            Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], self.nside, 1, hp.Alm.getlmax(dlm.size))
            del dlm
            tlm,elm,blm = self.get_unlensed_alm(idx)
            del blm
            T  = lenspyx.alm2lenmap(tlm, [Red, Imd], self.nside, 
                                    facres=self.facres, 
                                    verbose=False)
            del tlm
            Q, U  = lenspyx.alm2lenmap_spin([elm, None],[Red, Imd], 
                                            self.nside, 2, facres=self.facres,
                                            verbose=False)
            del (Red, Imd, elm)
            hp.write_map(fname,[T,Q,U],dtype=np.float64)
            self.vprint(f"CMB field cached: {idx}")         
            return [T,Q,U]
        
        
    def run_job(self):
        jobs = np.arange(self.nsim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Lensing map-{i} in processor-{mpi.rank}")
            NULL = self.get_lensed(i)
            del NULL

class ForeGround:

    def __init__(self,fg_str='s1d1',nside=2048):
        self.lb_table = Surveys().get_table_dataframe('LITEBIRD_V1')
        self.fg_dir = '/pscratch/sd/l/lonappan/DELL/FG'
        self.fg_str = fg_str
        self.nside = nside
        os.makedirs(self.fg_dir,exist_ok=True)
        print(f"Foregrounds: {self.fg_str}")
    
    @property
    def make(self):
        sky = pysm3.Sky(nside=self.nside, preset_strings=list(map(''.join, zip(*[iter(self.fg_str)]*2))))
        freq = self.lb_table.frequency.values
        for v in tqdm(freq,desc='Creating Foregrounds', unit='freq'):
            fname = os.path.join(self.fg_dir,f"{self.fg_str}_{int(v)}.fits")
            if not os.path.isfile(fname):
                maps = sky.get_emission(v * u.GHz)
                maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(v*u.GHz))
                hp.write_map(fname, maps.value,dtype=np.float64)
            else:
                print(f"{fname} already exists")
                continue

    def test_v(self,v,sky=None,plot=False):
        if sky is None:
            sky = pysm3.Sky(nside=256, preset_strings=list(map(''.join, zip(*[iter(self.fg_str)]*2))))
        maps = sky.get_emission(v * u.GHz)
        maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(v*u.GHz))
        alms_n = hp.map2alm(maps.value)
        tn = hp.alm2cl(alms_n[0])
        en = hp.alm2cl(alms_n[1])
        bn = hp.alm2cl(alms_n[2])
        del (maps,alms_n)
        
        fname = os.path.join(self.fg_dir,f"{self.fg_str}_{int(v)}.fits")
        maps = hp.read_map(fname,(0,1,2)) # type: ignore 
        alms_h = hp.map2alm(hp.ud_grade(maps,256))
        th = hp.alm2cl(alms_h[0])
        eh = hp.alm2cl(alms_h[1])
        bh = hp.alm2cl(alms_h[2])
        del (maps,alms_h)

        if plot:
            plt.figure(figsize=(8,8))
            plt.loglog(tn,label='tn')
            plt.loglog(th,label='to')
            plt.loglog(en,label='en')
            plt.loglog(eh,label='eo')
            plt.loglog(bn,label='bn')
            plt.loglog(bh,label='bo')
        else:
            return tn,th,en,eh,bn,bh

    @property
    def test(self):
        sky = pysm3.Sky(nside=256, preset_strings=list(map(''.join, zip(*[iter(self.fg_str)]*2))))
        freq = self.lb_table.frequency.values
        for v in freq:
            tn,to,en,eo,bn,bo = self.test_v(v,sky=sky) # type: ignore
            T = np.allclose(tn,to,rtol=1e-3,atol=1e-3)
            E = np.allclose(en,eo,rtol=1e-3,atol=1e-3)
            B = np.allclose(bn,bo,rtol=1e-3,atol=1e-3)

            if T and E and B:
                print(f"Frequency {v}GHz is consistent")
            else:
                print(f"Frequency {v}GHz is not consistent")


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-maps', dest='maps', action='store_true', help='map making')
    parser.add_argument('-noise', dest='noise', action='store_true', help='effective noise making')
    parser.add_argument('-beam', dest='beam', action='store_true', help='effective beam making')
    parser.add_argument('-fg', dest='fg', action='store_true', help='fg residuals making')
    parser.add_argument('-cmb', dest='cmb', action='store_true', help='mean CMB making')
    parser.add_argument('-lens', dest='lens', action='store_true', help='CMB Lensed')

    args = parser.parse_args()

    ini = args.inifile[0]

    print(ini)

    if args.lens:
        outfolder = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps_c/"
        nsim = 100
        cl_path = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CAMB/"
        scal_file = "BBSims_scal_dls.dat"
        pot_file = "BBSims_lenspotential.dat"
        phi='c'
        l = CMBLensed(outfolder, nsim, cl_path, scal_file, pot_file, phi,verbose=True)
        l.run_job()

    if args.maps:
        sim = SimExperimentFG.from_ini(ini)
        sim.run_job_mpi()

    if args.noise:
        sim = SimExperimentFG.from_ini(ini)
        sim.noise_mean_mpi()

    if args.beam:
        sim = SimExperimentFG.from_ini(ini)
        sim.beam_mean_mpi()
    
    if args.fg:
        sim = SimExperimentFG.from_ini(ini)
        sim.fg_res_mean_mpi()
    
    if args.cmb:
        sim = SimExperimentFG.from_ini(ini)
        sim.cmb_mean_mpi()

    mpi.barrier()