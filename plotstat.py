import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import seaborn as sns


class simStat:
    
    def __init__(self,sim_fg1,sim_fg2,fg1='s0d0',fg2='s1d1'):
        self.sim_fg1 = sim_fg1
        self.sim_fg2 = sim_fg2
        self.fg1 = fg1
        self.fg2 = fg2
    
    


class recStat:
    """
    Statistic class for the Reconstruction publication

    Parameters
    ----------
    rec_nofg : object : Reconstruction object with no foregrounds
    rec_fg1 : object : Reconstruction object with foregrounds
    """

    def __init__(self,rec_nofg,rec_fg1,rec_fg2,fg1='s0d0',fg2='s1d1'):
        self.rec_nofg = rec_nofg
        self.rec_fg1 = rec_fg1
        self.rec_fg2 = rec_fg2
        self.fg1 = fg1
        self.fg2 = fg2

    def plot_fg_impact(self,save=False):
        """
        Plot the impact of foregrounds on the reconstruction
        """
        rec_nofg = self.rec_nofg
        rec_fg1 = self.rec_fg1
        rec_fg2 = self.rec_fg2

        nofg_cl = rec_nofg.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
        fg1_cl = rec_fg1.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
        fg2_cl = rec_fg2.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
        fid = rec_fg1.bin_cell(rec_fg1.cl_pp*rec_fg1.Lfac)

        fig, axs = plt.subplots(2, 1,figsize=(9,9), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
        plt.setp(axs, xlim=(2,620))
        fig.subplots_adjust(hspace=0)


        axs[0].semilogy(rec_fg1.cl_pp*rec_fg1.Lfac,label='Fiducial',c='grey',lw=2)
        axs[0].semilogy(rec_nofg.Lfac*(rec_nofg.MCN0()/rec_nofg.response_mean()**2 ),label='NOFG-MCN0',c='b')
        axs[0].semilogy(rec_fg1.Lfac*(rec_fg1.MCN0()/rec_fg1.response_mean()**2 ),label=f'{self.fg1}-MCN0',c='r',)
        axs[0].semilogy(rec_fg2.Lfac*(rec_fg2.MCN0()/rec_fg2.response_mean()**2 ),label=f'{self.fg2}-MCN0',)
        axs[0].semilogy(rec_fg1.Lfac*(rec_nofg.MCN0()/rec_nofg.response_mean()**2 )/100,label='NOFG-MC Error')
        axs[0].semilogy(rec_fg1.Lfac*(rec_fg2.MCN0()/rec_fg2.response_mean()**2 )/100,label=f'{self.fg2} MC-Error')
        axs[0].semilogy(rec_nofg.Lfac*rec_nofg.mean_field_cl(),label='NOFG-MF',c='y')
        axs[0].semilogy(rec_fg2.Lfac*rec_fg2.mean_field_cl(),label=f'{self.fg2}-MF',c='g')
        axs[0].legend(ncol=2, fontsize=15)
        axs[0].set_ylim(1e-9,1e-5)
        axs[0].set_ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$',fontsize=20)

        axs[1].errorbar(rec_fg1.B,nofg_cl.mean(axis=0)/fid,yerr=nofg_cl.std(axis=0)/fid,label='NOFG',c='b',fmt='o')
        axs[1].errorbar(rec_fg1.B,fg2_cl.mean(axis=0)/fid,yerr=fg2_cl.std(axis=0)/fid,label=f'{self.fg2}',c='r',fmt='o')
        axs[1].set_ylim(-0.1,2.3)
        axs[1].legend(ncol=2, fontsize=15,loc='upper left')
        axs[1].axhline(1,c='k')
        axs[1].set_ylabel('$\\frac{C_L^{\phi\phi,rec}}{C_L^{\phi\phi,fid}}$',fontsize=20)
        axs[1].set_xlabel('$L$',fontsize=20)
        if save:
            plt.savefig('fg_impact.pdf', bbox_inches='tight',dpi=300)


    def plot_map_dif(self,idx=35,save=False,swap=False):
        """
        Plot the difference between the input and reconstructed maps
        """
        if swap:
            rec_fg1 = self.rec_fg2
        else:
            rec_fg1 = self.rec_fg1
        dir_ = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps/MASS"
        fname = os.path.join(dir_,f"phi_sims_{idx:04d}.fits")
        phi = hp.read_alm(fname)
        DfacL = np.arange(1025)
        Dfac = np.sqrt(DfacL*(DfacL+1))
        Dfac[:10] = 0
        dphi = hp.almxfl(phi,Dfac)
        input = hp.ma(hp.alm2map(dphi,512))
        input.mask = np.logical_not(rec_fg1.mask)
        output = hp.ma(rec_fg1.deflection_map(idx))
        output.mask = np.logical_not(rec_fg1.mask)

        plt.figure(figsize=(20,20))
        plt.subplots_adjust(wspace=5)
        hp.mollview(input,norm='hist',min=-.0005,max=.001,title='$\\sqrt{L(L+1)}\phi^{input}_{LM}$',sub=(1,2,1),notext=True)
        hp.mollview(output,norm='hist',min=-.0005,max=.001,title='$\\sqrt{L(L+1)}\phi^{WF,rec}_{LM}$',sub=(1,2,2),notext=True)
        if save:
            plt.savefig('def_comp.pdf', bbox_inches='tight',dpi=300)

    def SNR_impact(self):
        """
        Difference in SNR between the foreground and no foreground case
        """
        rec_nofg = self.rec_nofg
        rec_fg1 = self.rec_fg1
        rec_fg2 = self.rec_fg2
        SNR_nofg = rec_nofg.SNR_phi(rdn0=True)
        SNR_fg1 = rec_fg1.SNR_phi(rdn0=True)
        SNR_fg2 = rec_fg2.SNR_phi(rdn0=True)

        print(f"SNR NOFG: {SNR_nofg:.2f}")
        print(f"SNR FG1: {SNR_fg1:.2f} decreased by {(1-SNR_fg1/SNR_nofg)*100:.2f} %")
        print(f"SNR FG2: {SNR_fg2:.2f} decreased by {(1-SNR_fg2/SNR_nofg)*100:.2f} %")
    
    def plot_bin_corr_comp(self,which=1):
        if which == 1:
            rec = self.rec_fg1
        elif which == 2:
            rec = self.rec_fg2
        elif which == 0:
            rec = self.rec_nofg
        else:
            raise ValueError("which must be 0,1,2")

        f,(ax1,ax2, axcb) = plt.subplots(1,3, 
            gridspec_kw={'width_ratios':[1,1,0.08]},figsize=(10,5))
        ax1.get_shared_y_axes().join(ax1,ax2)
        g1 = sns.heatmap(rec.bin_corr(),cmap="coolwarm",cbar=False,ax=ax1)
        g1.set_ylabel('')
        g1.set_xlabel('')
        g1.set_title('MCN0')
        g2 = sns.heatmap(rec.bin_corr(rdn0=True),cmap="coolwarm",ax=ax2, cbar_ax=axcb)
        g2.set_ylabel('')
        g2.set_xlabel('')
        g2.set_yticks([])
        g2.set_title('RDN0')
 
        
