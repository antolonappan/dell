import numpy as np
import healpy as hp
import os
import mpi
import pickle as pl
from scipy.signal import savgol_filter


beam = 1/hp.gauss_beam(np.radians(.5),lmax=1500)


def get_noise(i):
    print(f"Reading MAP{i} from processor{mpi.rank}")
    cmb_path = os.path.join(os.environ['SCRATCH'],'S4BIRD','CMB_Lensed_Maps','CMB_SET1',f'cmb_sims_{i:04d}.fits')
    exp_path = os.path.join(os.environ['SCRATCH'],'S4BIRD','LiteBIRD','SIM_SET1_FG','Maps',f'exp_sims_{i:04d}.fits')
    cmb_map = hp.ud_grade(hp.read_map(cmb_path,(0,1,2)),512)
    exp_alm = hp.read_alm(exp_path,(1,2,3))
    hp.almxfl(exp_alm[0],beam,inplace=True)
    hp.almxfl(exp_alm[1],beam,inplace=True)
    hp.almxfl(exp_alm[2],beam,inplace=True)
    exp_map = hp.alm2map(exp_alm,512)
    mask_path = '/project/projectdirs/litebird/simulations/maps/lensing_project_paper/Masks/LB_Nside2048_fsky_0p8_binary.fits.gz'
    mask = hp.ud_grade(hp.read_map(mask_path),512)
    noise = exp_map - (cmb_map*mask)
    del (exp_map,cmb_map)
    nalm = hp.map2alm(noise)
    del noise
    ne = hp.alm2cl(nalm[1])
    nb = hp.alm2cl(nalm[2])
    del nalm
    return ne, nb


def get_mean_nl_mpi():
    ne,nb = get_noise(mpi.rank)
    if mpi.rank == 0:
        total_ne = np.zeros_like(ne)
        total_nb = np.zeros_like(nb)
    else:
        total_ne = None
        total_nb = None
        ne_mean = None
        nb_mean = None

    mpi.com.Reduce(ne,total_ne, op=mpi.mpi.SUM,root=0)
    mpi.com.Reduce(nb,total_nb, op=mpi.mpi.SUM,root=0)

    if mpi.rank == 0:
        ne_mean = total_ne/mpi.size
        nb_mean = total_nb/mpi.size
        fname = f'/project/projectdirs/litebird/simulations/maps/lensing_project_paper/DELL/LiteBIRD/ne_nb_{mpi.size}.pkl'
        pl.dump((ne_mean,nb_mean),open(fname,'wb'))

def smooth(y,d,c):
    yw = y.copy()*10**6
    yw[10:] = savgol_filter(yw[10:],d,c)
    return yw/10**6

def get_smooth_nl():
    fname = '/project/projectdirs/litebird/simulations/maps/lensing_project_paper/DELL/LiteBIRD/ne_nb_100.pkl'
    fname_f = '/project/projectdirs/litebird/simulations/maps/lensing_project_paper/DELL/LiteBIRD/ne_nb_100_smooth.pkl'
    ne,nb = pl.load(open(fname,'rb'))
    neh = smooth(ne,51,3)
    nbh = smooth(nb,51,3)
    pl.dump((neh,nbh),open(fname_f,'wb'))
    return neh,nbh

if __name__ == '__main__':
    get_smooth_nl()
