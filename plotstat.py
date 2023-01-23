import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import seaborn as sns
import pickle as pl



class simStat:
    
    def __init__(self,sim_fg1,sim_fg2,fg1='s0d0',fg2='s1d1'):
        self.sim_fg1 = sim_fg1
        self.sim_fg2 = sim_fg2
        assert self.sim_fg1.lmax == self.sim_fg2.lmax
        self.lmax = self.sim_fg1.lmax
        self.fg1 = fg1
        self.fg2 = fg2
    
    def plot_fg1(self):
        fname = f'../Data/paper/simFG1.pkl'
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
        plt.legend(ncol=3,fontsize=15)
        plt.xlabel("$\ell$",fontsize=20)
        plt.ylabel("$\frac{1}{b_\ell^2}\left(C_\ell + C_\ell^{FG\;res} + N_\ell \right)$ [$\mu K^2$]",fontsize=20)
        plt.tick_params(labelsize=20)
    
    def plot_fg2(self):
        l = np.arange(self.lmax+1)
        _,fg1_nl_e,fg1_nl_b = self.sim_fg1.noise_spectra(500)
        _,fg2_nl_e,fg2_nl_b = self.sim_fg2.noise_spectra(500)
        _,fg1_res_e,fg1_res_b = self.sim_fg1.fg_res_mean(500)
        _,fg2_res_e,fg2_res_b = self.sim_fg2.fg_res_mean(500)
        fg1_rnl_b = fg1_nl_b + fg1_res_b
        fg2_rnl_b = fg2_nl_b + fg2_res_b
        fg1_rnl_e = fg1_nl_e + fg1_res_e
        fg2_rnl_e = fg2_nl_e + fg2_res_e
        lfac = (l*(l+1))/(2*np.pi)
        plt.figure(figsize=(16, 8))
        fig, (ax1, ax2)  = plt.subplots(1, 2,figsize=(16, 8))

        ax1.loglog(self.sim_fg1.cl_len[1,:]*self.sim_fg1.Tcmb**2 ,c='k',lw=2,label="Fiducial EE")
        ax1.loglog(fg1_rnl_e*self.sim_fg1.Tcmb**2 ,label=f"{self.fg1}",c='r',lw=2,)
        ax1.loglog(fg2_rnl_e*self.sim_fg1.Tcmb**2 ,label=f"{self.fg2}",c='g',lw=2)
        ax1.legend(fontsize=15)
        ax1.set_xlabel("$\ell$",fontsize=20)
        ax1.set_ylabel("$\\frac{1}{b_\ell^2}\\left(C_\ell^{FG\;res} + N_\ell \\right)$ [$\mu K^2$]",fontsize=20)
        ax1.set_xlim(2,600)

        ax2.loglog(self.sim_fg1.cl_len[2,:]*self.sim_fg1.Tcmb**2 ,c='k',lw=2,label="Fiducial BB")
        ax2.loglog(fg1_rnl_b*self.sim_fg1.Tcmb**2 ,label=f"{self.fg1}",c='b',lw=2)
        ax2.loglog(fg2_rnl_b*self.sim_fg1.Tcmb**2 ,label=f"{self.fg2}",c='magenta',lw=2)
        ax2.legend(fontsize=15)
        ax2.set_xlabel("$\ell$",fontsize=20)
        ax2.set_xlim(2,600)
        






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
 
        
