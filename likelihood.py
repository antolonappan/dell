import emcee
import os
import numpy as np
from scipy.optimize import minimize
from getdist import plots, MCSamples
from quest import Reconstruction
import mpi
import pickle as pl
import matplotlib.pyplot as plt
import camb

class Likelihood_Alens:

    def __init__(self, lib_dir,rec_lib):
        self.lib_dir = os.path.join(lib_dir,'Likelihood_Alens')
        if mpi.rank==0:
            os.makedirs(self.lib_dir, exist_ok=True)
        self.rec = rec_lib
    
    @classmethod
    def from_ini(cls, ini_file):
        rec = Reconstruction.from_ini(ini_file)
        lib = rec.filt_lib.sim_lib.outfolder
        return cls(lib,rec)
    
    def theory_PP(self,alen):
        return self.rec.cl_pp * alen * self.rec.Lfac
    
    def theory_PP_binned(self,alen):
        return self.rec.bin_cell(self.theory_PP(alen))
    
    def data(self):
        return self.rec.get_qcl_wR_stat(n=400,ret='cl',n1=True,rdn0=True)
    
    def data_mean_spectra(self):
        return self.data().mean(axis=0)

    def data_covariance(self):
        return np.cov(self.data().T)

    def chisq(self,alen):
        theory = self.theory_PP_binned(alen)
        data = self.data_mean_spectra()
        cov = self.data_covariance()
        vect = theory - data
        return np.dot(vect, np.dot(np.linalg.inv(cov), vect.T))
    
    def lnlike(self,alen):
        return -0.5 * self.chisq(alen) # type: ignore   
     
    def lnprior(self,alen):
        if 0.5 < alen < 1.5:
            return 0.0
        return -np.inf
    
    def lnprob(self,alen):
        lp = self.lnprior(alen)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(alen)

    def MLE(self):
        np.random.seed(42)
        nll = lambda *args: -self.lnlike(*args)
        initial = np.array([1]) + 0.1 * np.random.randn(1)
        soln = minimize(nll, initial)
        if soln.success:
            return soln.x
        else:
            raise ValueError(soln.message)

    def sampler(self, nwalkers=32, nsteps=1000):
        fname = os.path.join(self.lib_dir,'samples.pkl' if nsteps==1000 else f'samples_{nsteps}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            ndim = 1
            pos = [[1] + 1e-1 * np.random.randn(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
            sampler.run_mcmc(pos, nsteps,progress=True)
            samples = sampler.get_chain(discard=int(.1*nsteps), thin=15, flat=True)
            pl.dump(samples,open(fname,'wb'))
            return samples
    
    def mcsamples(self,nsteps=1000):
        return MCSamples(samples=self.sampler(nsteps=nsteps),names=['alens'],labels=["A_{lens}"])
    
    def plot_posterior(self,nsteps=1000):
        g = plots.get_subplot_plotter(width_inch=5)
        g.plot_1d(self.mcsamples(nsteps), 'alens')
        plt.savefig(os.path.join(self.lib_dir,'posterior.png'))
    
    def Alens(self,nsteps=100,lim=1):
        return self.mcsamples(nsteps).getInlineLatex('alens',limit=lim)

class Likelihood_CAMB:

    def __init__(self,lib_dir,rec_lib):
        self.rec = rec_lib
        self.lib_dir = os.path.join(lib_dir,'Likelihood_CAMB')
        if mpi.rank==0:
            os.makedirs(self.lib_dir, exist_ok=True)
        self.lmax = len(self.rec.cl_pp) - 1
        self.pars = camb.CAMBparams()
        self.pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        self.pars.set_for_lmax(self.lmax, lens_potential_accuracy=0)

    def theory_PP(self,theta):
        omch2,alens = theta
        self.pars.set_cosmology(H0=67.5,omch2=omch2,Alens=alens)
        results = camb.get_results(self.pars)
        return results.get_lens_potential_cls(lmax=1024)[:,0]
    
    def theory_PP_binned(self,theta):
        return self.rec.bin_cell(self.theory_PP(theta))
    
    def data(self):
        return self.rec.get_qcl_wR_stat(n=400,ret='cl',n1=True,rdn0=True)
    
    def data_mean_spectra(self):
        return self.data().mean(axis=0)

    def data_covariance(self):
        return np.cov(self.data().T)

    def chisq(self,theta):
        theory = self.theory_PP_binned(theta)
        data = self.data_mean_spectra()
        cov = self.data_covariance()
        vect = theory - data
        return np.dot(vect, np.dot(np.linalg.inv(cov), vect.T))
    
    def lnlike(self,theta):
        return -0.5 * self.chisq(theta) # type: ignore   
     
    def lnprior(self,theta):
        omch2,alens = theta
        if 0.08 < omch2 < 0.15 and 0.5 < alens < 1.5:
            return 0.0
        return -np.inf
    
    def lnprob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    def MLE(self):
        np.random.seed(42)
        nll = lambda *args: -self.lnlike(*args)
        initial = np.array([0.12,1]) + 0.001 * np.random.randn(2)
        soln = minimize(nll, initial)
        return soln
        
    def sampler(self, nwalkers=32, nsteps=1000):
        fname = os.path.join(self.lib_dir,'samples.pkl' if nsteps==1000 else f'samples_{nsteps}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            ndim = 2
            pos = [np.array([.12,1]) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
            sampler.run_mcmc(pos, nsteps,progress=True)
            samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(samples,open('samples.pkl','wb'))
        return samples