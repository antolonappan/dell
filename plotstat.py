import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import seaborn as sns
import pickle as pl
import socket
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import emcee
from scipy.stats import gaussian_kde

datapath = '../Data/PlotData'
plotpath = '../Notebooks/plots'

os.makedirs(datapath,exist_ok=True)
os.makedirs(plotpath,exist_ok=True)

if socket.gethostname() == 'vmi401751.contaboserver.net':
    uselatex = True
    plt.rcParams['text.usetex']=True
    plt.rcParams['ytick.minor.visible'] =True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['font.size'] = '20'
    plt.rcParams['legend.fontsize'] = '18'
    plt.rcParams['legend.borderaxespad'] = '1'
    plt.rcParams['legend.numpoints'] = '1'
    plt.rcParams['figure.titlesize'] = 'medium'
    plt.rcParams['xtick.major.size'] = '10'
    plt.rcParams['xtick.minor.size'] = '6'
    plt.rcParams['xtick.major.width'] = '2'
    plt.rcParams['xtick.minor.width'] = '1'
    plt.rcParams['ytick.major.size'] = '10'
    plt.rcParams['ytick.minor.size'] = '6'
    plt.rcParams['ytick.major.width'] = '2'
    plt.rcParams['ytick.minor.width'] = '1'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.labelpad'] = '7.0'
    plt.rcParams['axes.formatter.limits']=-10,10
    plt.rcParams['xtick.labelsize'] = '20'
    plt.rcParams['ytick.labelsize'] = '20'
    plt.rcParams['axes.labelsize'] = '30'
    plt.rcParams['axes.labelsize'] = '30'
    plt.rcParams['xtick.major.pad']='10'
    plt.rcParams['xtick.minor.pad']='10'
    plt.rcParams['hatch.color'] = 'black'
    plt.rcParams['lines.dashed_pattern']=3, 1.5
else:
    uselatex = False
    
def find_density(arr):
    density = gaussian_kde(arr,.3)
    xs = np.linspace(min(arr),max(arr),1000)
    ds = density(xs)
    return xs,ds/max(ds)


class Alens_fit:
    """
    Class to fit for the Alens parameter

    Parameters
    ----------
    reco : object : Reconstruction object
    """
    
    def __init__(self,reco):
        self.rec = reco
        self.fid = reco.bin_cell(reco.cl_pp*reco.Lfac)
        self.spectra = self.spectra_()
        self.icov = self.icov_()
        
    def spectra_(self):
        """
        Get the spectra to fit for
        """
        return self.rec.get_qcl_wR_stat(400,rdn0=True,n1=True).mean(axis=0)
    
    def icov_(self):
        """
        Get the inverse covariance matrix
        """
        cov = np.cov(self.rec.get_qcl_wR_stat(400,rdn0=True,n1=True).T)
        return np.linalg.inv(cov)
    
    def chi_sq(self,alens):
        """
        Get the Chi^2
        """
        dcl = self.spectra - (alens*self.fid)
        return np.dot(dcl,np.dot(self.icov,dcl))
    
    def log_prior(self,theta):
        """
        Get the log prior
        """
        if 0.5 < theta < 1.5:
            return 0.0
        return -np.inf

    def log_likelihood(self,theta):
        """
        Get the log likelihood
        """
        return -0.5 * self.chi_sq(theta)

    def log_probability(self,theta):
        """
        Get the log probability
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    

    def get_samples(self):
        """
        Samples from the posterior
        """
        pos = [1] + 1e-1 * np.random.randn(64, 1)
        nwalkers,ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers,ndim,self.log_probability)
        sampler.run_mcmc(pos, 4000, progress=True)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return flat_samples

class crazymix:
    """
    Class to fit for the Alens parameter
    with mismatched simulations
    """

    def __init__(self,s1d1,other,idx=0,do_MC=False):
        self.s1d1 = s1d1
        self.other = other
        self.idx = idx
        self.B = self.s1d1.B
        self.do_MC = do_MC
        self.fiducial = self.other.bin_cell(self.other.cl_pp*self.other.Lfac)
        self.icov = self.icov_()
        self.spectra = self.get_spectra()

    def get_data(self):
        return self.s1d1.get_phi_cl(self.idx)

    def get_spectra(self):

        if self.do_MC:
            N0 = self.other.MCN0()
            correction =N0*0
        else:
            N0 = self.other.RDN0(self.idx)
            correction = ((N0/self.other.response_mean()**2)+self.other.cl_pp)/100
        cl = (self.get_data() - 
              self.other.N1 - 
              N0)/self.other.response_mean()**2

        return self.other.bin_cell((cl-correction)*self.s1d1.Lfac)

    def icov_(self):
        return np.linalg.inv(np.cov(self.other.get_qcl_wR_stat().T))

    def chi_sq(self,alens):
        dcl = self.spectra - (alens*self.fiducial)
        return np.dot(dcl,np.dot(self.icov,dcl))

    def log_prior(self,theta):
        if 0.5 < theta < 1.5:
            return 0.0
        return -np.inf

    def log_likelihood(self,theta,):
        return -0.5 * self.chi_sq(theta)

    def log_probability(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def get_samples(self):
        pos = [1] + 1e-1 * np.random.randn(64, 1)
        nwalkers,ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers,ndim,self.log_probability)
        sampler.run_mcmc(pos, 4000, progress=True)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return flat_samples
    

class Planck:
    """
    Class for Planck data
    """

    def __init__(self):
        datadir = '../Data/planck/'
        MV = np.loadtxt(datadir+'mv_nlkk.dat').T
        self.MV = {'L':MV[0],'N':MV[1]*(2/np.pi),'SN':MV[2]}
        PP = np.loadtxt(datadir+'pp_nlkk.dat').T
        self.PP = {'L':PP[0],'N':PP[1]*(2/np.pi),'SN':PP[2]}
        TT = np.loadtxt(datadir+'tt_nlkk.dat').T
        self.TT = {'L':TT[0],'N':TT[1]*(2/np.pi),'SN':TT[2]}

class simStat:
    
    def __init__(self,sim_fg1=None,sim_fg2=None,fg1='s0d0',fg2='s1d1'):
        self.sim_fg1 = sim_fg1
        self.sim_fg2 = sim_fg2
        if (self.sim_fg1 == self.sim_fg2 == None):
            pass
        else:
            assert self.sim_fg1.lmax == self.sim_fg2.lmax 
            self.lmax = self.sim_fg1.lmax
        self.fg1 = fg1
        self.fg2 = fg2
    
    def plot_fg1(self,save=False):
        fname = os.path.join(datapath,'simFG1.pkl')
        if os.path.isfile(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
            stat1 = data['stat1']
            stat2 = data['stat2']
        else:
            data = {}
            stat1 = self.sim_fg1.cl_stat(100)
            stat2 = self.sim_fg2.cl_stat(100)
            data['stat1'] = stat1
            data['stat2'] = stat2
            data['fid_ee'] = self.sim_fg1.cl_len[1,:]*self.sim_fg1.Tcmb**2
            data['fid_bb'] = self.sim_fg1.cl_len[2,:]*self.sim_fg1.Tcmb**2
            data['l'] = np.arange(self.lmax+1)
            data['fg1'] = self.fg1
            data['fg2'] = self.fg2
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')
        plt.figure(figsize=(8,8))
        plt.loglog(data['fid_ee'] ,c='k',ls='--',lw=2,label="Fiducial EE")
        plt.loglog(data['fid_bb'] ,c='k',ls='-.',lw=2,label="Fiducial BB")
        plt.loglog(stat1[0],label=f"{data['fg1']} EE",c='r',lw=2)
        plt.fill_between(data['l'],stat1[0]-(stat1[1]),stat1[0]+stat1[1],color='r',alpha=0.5)
        plt.loglog(stat1[2],label=f"{data['fg1']} BB",c='b',lw=2)
        plt.fill_between(data['l'],stat1[2]-stat1[3],stat1[2]+stat1[3],color='b',alpha=0.5)
        plt.loglog(stat2[0],label=f"{data['fg2']} EE",c='g',lw=2)
        plt.fill_between(data['l'],stat2[0]-(stat2[1]),stat2[0]+stat2[1],color='g',alpha=0.5)
        plt.loglog(stat2[2],label=f"{data['fg2']} BB",c='magenta',lw=2)
        plt.fill_between(data['l'],stat2[2]-stat2[3],stat2[2]+stat2[3],color='magenta',alpha=0.5)
        plt.xlim(2,800)
        plt.ylim(1e-7,10)
        plt.legend(ncol=3,fontsize=16)
        plt.xlabel("$\ell$",fontsize=25)
        plt.ylabel("$\\frac{1}{b_\ell^2}\\left(C^{fid}_\ell + C_\ell^{fg_{res}} + N_\ell \\right)$ [$\mu K^2$]",fontsize=25)
        plt.tick_params(labelsize=20)
        if save:
            plt.savefig('plots/simFG1.pdf',bbox_inches='tight',dpi=300)
    
    def plot_fg2(self,save=False):
        fname = os.path.join(datapath,'simFG2.pkl')
        if not os.path.isfile(fname):
            data = {}
            _,fg1_nl_e,fg1_nl_b = self.sim_fg1.noise_spectra(500)
            _,fg2_nl_e,fg2_nl_b = self.sim_fg2.noise_spectra(500)
            _,fg1_res_e,fg1_res_b = self.sim_fg1.fg_res_mean(500)
            _,fg2_res_e,fg2_res_b = self.sim_fg2.fg_res_mean(500)
            data['fg1_rnl_b'] = (fg1_nl_b + fg1_res_b)*self.sim_fg1.Tcmb**2
            data['fg2_rnl_b'] = (fg2_nl_b + fg2_res_b)*self.sim_fg1.Tcmb**2
            data['fg1_rnl_e'] = (fg1_nl_e + fg1_res_e)*self.sim_fg1.Tcmb**2
            data['fg2_rnl_e'] = (fg2_nl_e + fg2_res_e)*self.sim_fg1.Tcmb**2
            data['fid_ee'] = self.sim_fg1.cl_len[1,:]*self.sim_fg1.Tcmb**2
            data['fid_bb'] = self.sim_fg1.cl_len[2,:]*self.sim_fg1.Tcmb**2
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')
        else:
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')


        fg1_rnl_b = data['fg1_rnl_b']
        fg2_rnl_b = data['fg2_rnl_b']
        fg1_rnl_e = data['fg1_rnl_e']
        fg2_rnl_e = data['fg2_rnl_e']

        plt.figure(figsize=(16, 8))
        fig, (ax1, ax2)  = plt.subplots(1, 2,figsize=(16, 8))

        ax1.loglog(data['fid_ee'] ,c='k',lw=2,label="Signal EE")
        ax1.loglog(fg1_rnl_e ,label=f"{self.fg1}",c='r',lw=2,)
        ax1.loglog(fg2_rnl_e ,label=f"{self.fg2}",c='g',lw=2)
        ax1.legend(fontsize=20)
        ax1.set_xlabel("$\ell$",fontsize=25)
        ax1.set_ylabel("$\\frac{1}{b_\ell^2}\\left(C_\ell^{fg_{res}} + N_\ell \\right)$ [$\mu K^2$]",fontsize=25)
        ax1.set_xlim(2,600)
        ax1.set_ylim(1e-6,1e-1)

        ax2.loglog(data['fid_bb'] ,c='k',lw=2,label="Signal BB")
        ax2.loglog(fg1_rnl_b ,label=f"{self.fg1}",c='r',lw=2)
        ax2.loglog(fg2_rnl_b ,label=f"{self.fg2}",c='g',lw=2)
        ax2.legend(fontsize=20)
        ax2.set_xlabel("$\ell$",fontsize=25)
        ax2.set_ylim(1e-7,1e-2)
        ax2.set_xlim(2,600)

        if save:
            plt.savefig(os.path.join(plotpath,'simFG2.pdf'),bbox_inches='tight',dpi=300)







class recStat:
    """
    Statistic class for the Reconstruction publication

    Parameters
    ----------
    rec_nofg : object : Reconstruction object with no foregrounds
    rec_fg1 : object : Reconstruction object with foregrounds
    """

    def __init__(self,rec_nofg=None,rec_fg1=None,rec_fg2=None,rec_fg3=None,fg1='s0d0',fg2='s1d1'):
        self.rec_nofg = rec_nofg
        self.rec_fg1 = rec_fg1
        self.rec_fg2 = rec_fg2
        self.rec_fg3 = rec_fg3
        self.fg1 = fg1
        self.fg2 = fg2

    def plot_fg_impactNew(self,save=False,planck=True,logy=False,shift=0):
        """
        Plot the impact of foregrounds on the reconstruction
        """

        fname = os.path.join(datapath,'recFG.pkl')
        if os.path.isfile(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            rec_nofg = self.rec_nofg
            rec_fg1 = self.rec_fg1
            rec_fg2 = self.rec_fg2
            rec_fg3 = self.rec_fg3
            data = {}
            data['nofg_cl'] = rec_nofg.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
            data['fg2_cl'] = rec_fg2.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
            data['fid'] = rec_fg1.bin_cell(rec_fg1.cl_pp*rec_fg1.Lfac)
            data['fidm'] = rec_fg1.cl_pp*rec_fg1.Lfac
            data['NOFG-MCN0'] = rec_nofg.Lfac*(rec_nofg.RDN0_mean()/rec_nofg.response_mean()**2 )
            data['fg1-MCN0'] = rec_fg1.Lfac*(rec_fg1.RDN0_mean()/rec_fg1.response_mean()**2 )
            data['fg2-MCN0'] = rec_fg2.Lfac*(rec_fg2.RDN0_mean()/rec_fg2.response_mean()**2 )
            data['fg3-MCN0'] = rec_fg3.Lfac*(rec_fg3.RDN0_mean()/rec_fg3.response_mean()**2 )
            data['NOFG-MCER'] = rec_fg1.Lfac*(rec_nofg.MCN0()/rec_nofg.response_mean()**2 )/100
            data['fg2-MCER'] = rec_fg1.Lfac*(rec_fg2.MCN0()/rec_fg2.response_mean()**2 )/100
            data['NOFG-MF'] = rec_nofg.Lfac*rec_nofg.mean_field_cl()
            data['fg2-MF'] = rec_fg2.Lfac*rec_fg2.mean_field_cl()
            data['B'] = rec_fg1.B
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')




        nofg_cl = data['nofg_cl']
        fg2_cl = data['fg2_cl']
        fid = data['fid']

        fig, axs = plt.subplots(2, 1,figsize=(9,9), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
        plt.setp(axs, xlim=(2,690))
        fig.subplots_adjust(hspace=0)

        axs[0].semilogy(data['fidm'],label='Signal',c='grey',lw=2)
        axs[0].semilogy(data['NOFG-MCN0'], c='g', label="$\\textbf{No FG}$" if uselatex else "No FG",lw=2)
        axs[0].semilogy(data['fg1-MCN0'], c='b', label="$\\textbf{s0d0}$" if uselatex else "s0d0",lw=2)
        axs[0].semilogy(data['fg2-MCN0'], c='r', label="$\\textbf{s1d1}$" if uselatex else "s1d1",lw=2)

        axs[0].semilogy(data['NOFG-MCER']+(data['fidm']/100),c='g',ls=':',lw=3)
        axs[0].semilogy(data['fg2-MCER']+(data['fidm']/100),c='r',ls=':',lw=3)

        axs[0].semilogy(data['NOFG-MF'],c='g',ls='-.')
        axs[0].semilogy(data['fg2-MF'],c='r',ls='-.')
        if planck:
            axs[0].semilogy((Planck().MV['N'])[:len(data['fg1-MCN0'])],label='Planck(MV)')
        if logy:
            axs[0].semilogx()
        #axs[0].legend(ncol=3, fontsize=16,frameon=False)
        axs[0].set_ylim(1e-9,1e-5)
        axs[0].set_ylabel('$\\frac{L^2 (L + 1)^2}{2\pi} C_L^{\phi\phi}$',fontsize=25)
        legend1 = axs[0].legend(loc='upper left', fontsize=15)


        legend2_elements = [
            plt.Line2D([0], [0], color='black', lw=2, label="$\langle N_L^{(0),RD} \\rangle$"),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='$C_L^{MC}$'),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='-.', label='$C_L^{MF}$')
        ]


        legend2 = axs[0].legend(handles=legend2_elements, loc='upper right', fontsize=15, frameon=False)


        axs[0].add_artist(legend1)
        axs[0].add_artist(legend2)



        axs[1].errorbar(data['B']+shift,fg2_cl.mean(axis=0)/fid,yerr=fg2_cl.std(axis=0)/fid,label=f'{self.fg2}',c='r',fmt='o')
        axs[1].errorbar(data['B'],nofg_cl.mean(axis=0)/fid,yerr=nofg_cl.std(axis=0)/fid,label='NOFG',c='g',fmt='o')
        axs[1].set_ylim(-0.1,2.3)
        #axs[1].legend(ncol=2, fontsize=15,loc='upper right')
        axs[1].axhline(1,c='k')
        axs[1].set_ylabel('$\\frac{C_L^{\phi\phi,rec}}{C_L^{\phi\phi,signal}}$',fontsize=25)
        axs[1].set_xlabel("$L$",fontsize=25)
        if save:
            plt.savefig(os.path.join(plotpath,'recFG.pdf'), bbox_inches='tight',dpi=300)




    def plot_map_dif(self,idx=0,save=False,choose='best'):
        """
        Plot the difference between the input and reconstructed maps
        """
        fnamei = os.path.join(datapath,'phi_input_0.fits')
        fnameo = os.path.join(datapath,'phi_output_0.fits')
        
        rec_fg3 = self.rec_fg3

        if os.path.exists(fnamei):
            inputk = hp.read_map(fnamei)
            print('Input Map Loaded from file')
        else:
            inputk = rec_fg3.get_input_phi_sim(0,True,80)
            hp.write_map(fnamei,inputk)
            print('Input Map Saved to file')
        
        if os.path.exists(fnameo):
            output = hp.read_map(fnameo)
            print('Output Map Loaded from file')
        else:
            output = rec_fg3.wf_phi(0,True,80)
            hp.write_map(fnameo,output)
            print('Output Map Saved to file')
        
        rcomb = [[68,81],[77,66],[79,92],[95,55],[88,51]]
        if (choose == 'best') or None:
            which =  0
        elif choose == 'rand':
            which = np.random.choice(np.arange(len(rcomb)))
        elif type(choose) == int:
            which = choose
        r1,r2 = rcomb[which]
        res = 5
        xsize=200
        gs = gridspec.GridSpec(1, 2,wspace=0.1)
        plt.figure(figsize=(15,15))
        ax = plt.subplot(gs[0, 0])
        hp.gnomview(inputk,reso=res,rot=[r1,r2],norm='hist',title='',xsize=xsize,notext=True,hold=True)
        plt.title('$\kappa_{LM}^\\texttt{Input}$' if uselatex else '$\kappa_{LM}^{Input}$',fontsize=20)
        #plt.text(-.16,-.135,"$Resolution = 5^\prime/pixel, 200\\times200\;pixel$",rotation=90,fontsize=18)
        ax = plt.subplot(gs[0, 1])
        hp.gnomview(output,reso=res,rot=[r1,r2],norm='hist',xsize=xsize,title='Output',notext=True,hold=True)
        plt.title('$\kappa_{LM}^\\texttt{Output}$' if uselatex else '$\kappa_{LM}^{Output}$',fontsize=18)
        #ax = plt.subplot(gs[0, 2])
        #hp.gnomview(inputk-output,reso=res,rot=[r1,r2],norm='hist',xsize=xsize,title='Output',notext=True,hold=True)
        #plt.title('$\kappa_{LM}^\\texttt{Input-Output}$',fontsize=18)
        if save:
            plt.savefig(os.path.join(plotpath,'recMaps.pdf'), bbox_inches='tight',dpi=300)



    def plot_SNR_impact(self,save=False,color='gold'):
        """
        Difference in SNR between the foreground and no foreground case
        """
        fname = os.path.join(datapath,'snr.pkl')
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('SNR Loaded from file')
        else:
            data = {}
            rec_nofg = self.rec_nofg
            rec_fg1 = self.rec_fg1
            rec_fg2 = self.rec_fg2
            rec_fg3 = self.rec_fg3
            data['NOFG-SNR'] = rec_nofg.SNR_phi(rdn0=True)
            data['fg1-SNR'] = rec_fg1.SNR_phi(rdn0=True)
            data['fg2-SNR'] = rec_fg2.SNR_phi(rdn0=True)
            data['fg3-SNR'] = rec_fg3.SNR_phi(rdn0=True)
            pl.dump(data,open(fname,'wb'))
            print('SNR Saved to file')
        
        SNR_nofg =  data['NOFG-SNR']
        SNR_fg1 = data['fg1-SNR']
        SNR_fg2 = data['fg2-SNR']
        SNR_fg3 = data['fg3-SNR']
        #cases = ['Planck(pol)','Planck(MV)','No FG', 's0d0', 's1d1']
        if uselatex:
            cases = ['$\\textbf{No FG}$', '$\\textbf{s0d0}$','$\\textbf{s1d1}$',"$\\textbf{s1d1}$ \n ($f_{sky}=0.9$)",'$\\textsc{Planck}$'+"\n $(Pol)$"]#,'Planck(MV)']
        else:
            cases = ['No FG', 's0d0', 's1d1',"$s1d1$ \n ($f_{sky}=0.9$)",'Planck'+"\n $(Pol)$"]
        pl_pol = 9
        pl_mv = 40
        pl_pol_npipe = pl_pol*.2+pl_pol
        pl_mv_npipe = pl_mv*.2+pl_mv
        snr = [SNR_nofg,SNR_fg1,SNR_fg2,SNR_fg3,pl_pol_npipe]#,pl_mv_npipe]
        plt.figure(figsize=(8, 6))
        
        plot = plt.barh(cases[::-1],snr[::-1],color=color)
        
        for i,value in enumerate(plot):
            width = value.get_width()
            plt.text(width + 5, value.get_y()+.25,f"${width:.2f}$", ha='center', va='bottom', fontsize=25)

        

        plt.xlabel("Signal to Noise Ratio",fontsize=20) 
        plt.xlim(0,60)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15,rotation=45)
        if save:
            plt.savefig(os.path.join(plotpath,'SNR_impact_v2.pdf'), bbox_inches='tight',dpi=300)      
    
    def plot_qcl_stat(self,save=False):
        fname = os.path.join(datapath,'recQCL.pkl')
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            data = {}
            data['stat']= self.rec_fg3.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
            data['fid'] = self.rec_fg3.cl_pp*self.rec_fg3.Lfac
            data['mcn0'] = self.rec_fg3.Lfac*(self.rec_fg3.MCN0()/self.rec_fg3.response_mean()**2 )
            data['mcn1'] = self.rec_fg3.Lfac*self.rec_fg3.N1
            data['mf'] = self.rec_fg3.Lfac*self.rec_fg3.mean_field_cl()
            data['B'] = self.rec_fg3.B
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')

        
        stat = data['stat']
        plt.figure(figsize=(8,7))
        plt.loglog(data['fid'],label='$\\texttt{Signal}$' if uselatex else 'Signal',c='grey',lw=2)
        plt.loglog(data['mcn0'],label='$N_L^{(0)}$',c='r')
        plt.loglog(data['mcn1'],label='$N_L^{(1)}$',c='g')
        plt.loglog(data['mf'],label='$C_L^{MF}$',c='b')
        plt.errorbar(data['B'],stat.mean(axis=0),yerr=stat.std(axis=0),fmt='o',c='k',ms=6,capsize=2,label='$\\texttt{Reconstructed}$' if uselatex else 'Reconstructed')
        plt.xlim(2,600)
        plt.legend(ncol=2, fontsize=20)
        plt.xlabel('$L$',fontsize=25)
        plt.ylabel('$\\frac{L^2 (L + 1)^2}{2\pi} C_L^{\phi\phi}$',fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if save:
            plt.savefig(os.path.join(plotpath,'recQCL.pdf'), bbox_inches='tight',dpi=300)
    
    def plot_bin_corr_comp(self,which=1,save=False):
        fname = os.path.join(datapath,'bin_corr.pkl')
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            data = [None,None,None]
            data[0] = {}
            data[0]['mcn0'] = self.rec_nofg.bin_corr()
            data[0]['rdn0'] = self.rec_nofg.bin_corr(rdn0=True)
            data[1] = {}
            data[1]['mcn0'] = self.rec_fg1.bin_corr()
            data[1]['rdn0'] = self.rec_fg1.bin_corr(rdn0=True)
            data[2] = {}
            data[2]['mcn0'] = self.rec_fg2.bin_corr()
            data[2]['rdn0'] = self.rec_fg2.bin_corr(rdn0=True)
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')

        f,(ax1,ax2, axcb) = plt.subplots(1,3, gridspec_kw={'width_ratios':[1,1,0.08]},figsize=(10,5))

        ax1.get_shared_y_axes().join(ax1,ax2)
        g1 = sns.heatmap(data[which]['mcn0'],cmap="copper",cbar=False,ax=ax1)
        g1.set_ylabel('$b$',fontsize=25)
        g1.set_xlabel('$b$',fontsize=25)
        g1.set_title('$N_L^{(0),MC}$',fontsize=25)
        g2 = sns.heatmap(data[which]['rdn0'],cmap="copper",ax=ax2, cbar_ax=axcb)
        #g2.set_ylabel('b')
        g2.set_xlabel('$b$',fontsize=25)
        g2.set_yticks([])
        g2.set_title('$N_L^{(0),RD}$',fontsize=25)
        if save:
            plt.savefig(os.path.join(plotpath,f'recCor{which}.pdf'), bbox_inches='tight',dpi=300)

    
    def plot_planck_comparsion(self,save=False):
        fname = os.path.join(datapath,'recQCL.pkl')
        data = pl.load(open(fname,'rb'))
        
        plt.figure(figsize=(7,7))
        plt.loglog((Planck().PP['N'])[:len(data['mcn0'])],label='$\\textsc{Planck}\\texttt{(Pol)}$' if uselatex else "$Planck(Pol)$" ,lw=3)
        plt.loglog(data['mcn0'],label='$\\textsc{LiteBIRD}\\texttt{(EB)}$' if uselatex else "$LiteBIRD(EB)$",c='r',lw=3)
        plt.loglog(data['fid'],label='Signal',c='grey',lw=3)
        plt.xlim(2,600)
        plt.ylabel('$\\frac{L^2 (L + 1)^2}{2\pi} N_L^{(0),MC}$',fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('L',fontsize=25)
        plt.legend(ncol=2, fontsize=20)
        if save:
             plt.savefig(os.path.join(plotpath,f'planck_comp.pdf'), bbox_inches='tight',dpi=300)
    
    def plot_Alens_box(self,save=False):
        fname = os.path.join(datapath,'Alens_samps.pkl')
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            
            data = {}
            rec1 = self.rec_nofg
            rec2 = self.rec_fg1
            rec3 = self.rec_fg2
            data['nofg_samp'] = Alens_fit(rec1).get_samples().reshape(-1)
            data['fg1_samp'] = Alens_fit(rec2).get_samples().reshape(-1)
            data['fg2_samp'] = Alens_fit(rec3).get_samples().reshape(-1)
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')

        nofg_samp = data['nofg_samp']
        fg1_samp = data['fg1_samp']
        fg2_samp = data['fg2_samp']
        data_to_plot = [nofg_samp, fg1_samp, fg2_samp]
        if uselatex:
            labels = ['$\\textbf{No FG}$', '$\\textbf{s0d0}$', '$\\textbf{s1d1}$']
        else:
            labels = ['No FG', 's0d0', 's1d1']

        plt.figure(figsize=(6, 6))
        sns.boxplot(data=data_to_plot)
        plt.xticks(range(len(labels)),labels,fontsize=25)
        plt.ylabel('$A_{\mathrm{lens}}$',fontsize=25)
        plt.axhline(1,color='r',ls='--',lw=3)
        plt.yticks(fontsize=25)
        if save:
            plt.savefig(os.path.join(plotpath,f'Alens_box.pdf'), bbox_inches='tight',dpi=300)
    
    
    
    def plot_KS_pvalue(self,save=False):
        fname = os.path.join(plotpath,'pvalue.pdf')
        pvalue = pl.load(open(os.path.join(datapath,'pvalue.pkl'),'rb'))
        s0d0 = pvalue['s0d0']
        s1d1 = pvalue['s1d1']
        B = pvalue['b']
        plt.figure(figsize=(6,6))
        plt.plot(B,s0d0,label='$\\textbf{s0d0}$' if uselatex else "s0d0",marker='o',c='C10')
        plt.plot(B,s1d1,label='$\\textbf{s1d1}$' if uselatex else "s1d1",marker='o',c='C3')
        plt.axhline(0.05,color='k',ls='--',label='$0.05$',lw=3)
        plt.legend(fontsize=15)
        plt.ylabel('$\\mathrm{KS} \;\; p-\\mathrm{value}$' if uselatex else "KS p-value",fontsize=25)
        plt.xlabel('$L$',fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        if save:
            plt.savefig(fname, bbox_inches='tight',dpi=300)
    
    def plot_mismatching_alens(self,save=False,do_MC=False):
        fname = os.path.join(datapath,f'mismatch_samples{do_MC}.pkl')
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            rec1 = self.rec_nofg
            rec2 = self.rec_fg1
            rec3 = self.rec_fg2
            data ={}
            data['nofg'] = crazymix(rec3,rec1,100,do_MC).get_samples().reshape(-1)
            data['fg1'] = crazymix(rec3,rec2,100,do_MC).get_samples().reshape(-1)
            data['fg2'] = crazymix(rec3,rec3,100,do_MC).get_samples().reshape(-1)
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')

        snofg = data['nofg']
        sfg1 = data['fg1']
        sfg2 = data['fg2']

        smm1=np.mean(snofg)
        smm2=np.mean(sfg1)
        smm3=np.mean(sfg2)
        
        plt.figure(figsize=(8,6))
        plt.plot(*find_density(snofg),lw=3,label='$\\textbf{No FG}$'if uselatex else "No FG")
        plt.plot(*find_density(sfg1),lw=3,label='$\\textbf{s0d0}$' if uselatex else "s0d0")
        plt.plot(*find_density(sfg2),lw=3,label='$\\textbf{s1d1}$' if uselatex else "s1d1")
        plt.text(smm1-.013,1.02,f"{smm1:.2f}",fontsize=25)
        plt.text(smm2-.013,1.02,f"{smm2:.2f}",fontsize=25)
        plt.text(smm3-.013,1.02,f"{smm3:.2f}",fontsize=25)
        plt.xlabel('$A_\mathrm{lens}$',fontsize=25)
        plt.ylim(0,1.2)
        plt.legend(fontsize=18,loc='lower center',bbox_to_anchor=(0.6, 0.4))
        plt.xticks(fontsize=25)
        plt.yticks([])
        if save:
            plt.savefig(os.path.join(plotpath,f'MM_alens{do_MC}.pdf'), bbox_inches='tight',dpi=300)






class snrStat:
    def __init__(self,fg0=None,fg1=None,fg2=None):
        self.fg0 = fg0
        self.fg1 = fg1
        self.fg2 = fg2

    def table_total_snr(self):
        print('SNR total')
        print('NOFG :',self.fg0.snr_total())
        print('s0d0 :',self.fg1.snr_total())
        print('s1d1 :',self.fg2.snr_total())
    
    def table_tomo_snr(self):
        print('SNR tomo')
        print('NOFG :',self.fg0.snr_tomo())
        print('s0d0 :',self.fg1.snr_tomo())
        print('s1d1 :',self.fg2.snr_tomo())

    def plot_total_survey(self,save=False):
        plt.plot(self.fg0.ell,self.fg0.snr_tot_survey(),label='NOFG')
        plt.plot(self.fg1.ell,self.fg1.snr_tot_survey(),label='s0d0')
        plt.plot(self.fg2.ell,self.fg2.snr_tot_survey(),label='s1d1')
        plt.legend()

    def plot_tomo_survey(self,save=False):
        plt.plot(self.fg0.ell,self.fg0.snr_tomo_survey(),label='NOFG')
        plt.plot(self.fg1.ell,self.fg1.snr_tomo_survey(),label='s0d0')
        plt.plot(self.fg2.ell,self.fg2.snr_tomo_survey(),label='s1d1')
        plt.legend()
    def table_isw_snr(self):
        print('SNR ISW')
        print('NOFG :',self.fg0.snr_iswphi())
        print('s0d0 :',self.fg1.snr_iswphi())
        print('s1d1 :',self.fg2.snr_iswphi())


