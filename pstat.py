import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import seaborn as sns

class Stat:

    def __init__(self,rec_nofg,rec_fg):
        self.rec_nofg = rec_nofg
        self.rec_fg = rec_fg    

    def plot_fg_impact(self,save=False):
        rec_nofg = self.rec_nofg
        rec_fg = self.rec_fg

        nofg_cl = rec_nofg.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
        fg_cl = rec_fg.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
        fid = rec_fg.bin_cell(rec_fg.cl_pp*rec_fg.Lfac)

        fig, axs = plt.subplots(2, 1,figsize=(9,9), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
        plt.setp(axs, xlim=(2,620))
        fig.subplots_adjust(hspace=0)


        axs[0].semilogy(rec_fg.cl_pp*rec_fg.Lfac,label='Fiducial',c='grey',lw=2)
        axs[0].semilogy(rec_nofg.Lfac*(rec_nofg.MCN0()/rec_nofg.response_mean()**2 ),label='NOFG-MCN0',c='b')
        axs[0].semilogy(rec_fg.Lfac*(rec_fg.MCN0()/rec_fg.response_mean()**2 ),label='FG-MCN0',c='r',)
        axs[0].semilogy(rec_fg.Lfac*(rec_nofg.MCN0()/rec_nofg.response_mean()**2 )/100,label='NOFG-MC Error')
        axs[0].semilogy(rec_fg.Lfac*(rec_fg.MCN0()/rec_fg.response_mean()**2 )/100,label='FG MC-Error')
        axs[0].semilogy(rec_nofg.Lfac*rec_nofg.mean_field_cl(),label='NOFG-MF',c='y')
        axs[0].semilogy(rec_fg.Lfac*rec_fg.mean_field_cl(),label='FG-MF',c='g')
        axs[0].legend(ncol=3, fontsize=15)
        axs[0].set_ylim(1e-9,1e-5)
        axs[0].set_ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$',fontsize=20)


        axs[1].errorbar(rec_fg.B,fg_cl.mean(axis=0)/fid,yerr=fg_cl.std(axis=0)/fid,label='FG',c='r',fmt='o')
        axs[1].errorbar(rec_fg.B,nofg_cl.mean(axis=0)/fid,yerr=nofg_cl.std(axis=0)/fid,label='NOFG',c='b',fmt='o')
        axs[1].set_ylim(-0.1,2.3)
        axs[1].legend(ncol=2, fontsize=15,loc='upper left')
        axs[1].axhline(1,c='k')
        axs[1].set_ylabel('$\\frac{C_L^{\phi\phi,rec}}{C_L^{\phi\phi,fid}}$',fontsize=20)
        axs[1].set_xlabel('$L$',fontsize=20)
        if save:
            plt.savefig('fg_impact.pdf', bbox_inches='tight',dpi=300)


    def plot_map_dif(self,idx=35,save=False):
        rec_fg = self.rec_fg
        dir_ = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps/MASS"
        fname = os.path.join(dir_,f"phi_sims_{idx:04d}.fits")
        phi = hp.read_alm(fname)
        DfacL = np.arange(1025)
        Dfac = np.sqrt(DfacL*(DfacL+1))
        Dfac[:10] = 0
        dphi = hp.almxfl(phi,Dfac)
        input = hp.ma(hp.alm2map(dphi,512))
        input.mask = np.logical_not(rec_fg.mask)
        output = hp.ma(rec_fg.deflection_map(idx))
        output.mask = np.logical_not(rec_fg.mask)

        plt.figure(figsize=(20,20))
        plt.subplots_adjust(wspace=5)
        hp.mollview(input,norm='hist',min=-.0005,max=.001,title='$\\sqrt{L(L+1)}\phi^{input}_{LM}$',sub=(1,2,1),notext=True)
        hp.mollview(output,norm='hist',min=-.0005,max=.001,title='$\\sqrt{L(L+1)}\phi^{WF,rec}_{LM}$',sub=(1,2,2),notext=True)
        if save:
            plt.savefig('def_comp.pdf', bbox_inches='tight',dpi=300)

    def SNR_impact(self):
        rec_nofg = self.rec_nofg
        rec_fg = self.rec_fg
        SNR_nofg = rec_nofg.SNR_phi(rdn0=True)
        SNR_fg = rec_fg.SNR_phi(rdn0=True)
        print(f"SNR NOFG: {SNR_nofg:.2f}")
        print(f"SNR FG: {SNR_fg:.2f}")
        print(f"SNR decreased: {(1-SNR_fg/SNR_nofg)*100:.2f} %")
    
    def plot_bin_corr_comp(self,fg=False):
        """
        TODO
        """
        if fg:
            rec = self.rec_fg
        else:
            rec = self.rec_nofg
        fig, (ax1,ax2) = plt.subplots(1, 2)
        ax1 = sns.heatmap(rec.bin_corr())
        ax2 = sns.heatmap(rec.bin_corr(rdn0=True))
        ax1.set_title('MCN0')
        ax2.set_title('RDN0')


