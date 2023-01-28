import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import seaborn as sns
import pickle as pl
import socket

if socket.gethostname() == 'vmi401751.contaboserver.net':
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
        fname = '../Data/paper/simFG1.pkl'
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
        fname = '../Data/paper/simFG2.pkl'
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

        ax1.loglog(data['fid_ee'] ,c='k',lw=2,label="Fiducial EE")
        ax1.loglog(fg1_rnl_e ,label=f"{self.fg1}",c='r',lw=2,)
        ax1.loglog(fg2_rnl_e ,label=f"{self.fg2}",c='g',lw=2)
        ax1.legend(fontsize=20)
        ax1.set_xlabel("$\ell$",fontsize=25)
        ax1.set_ylabel("$\\frac{1}{b_\ell^2}\\left(C_\ell^{fg_{res}} + N_\ell \\right)$ [$\mu K^2$]",fontsize=25)
        ax1.set_xlim(2,600)
        ax1.set_ylim(1e-6,1e-1)

        ax2.loglog(data['fid_bb'] ,c='k',lw=2,label="Fiducial BB")
        ax2.loglog(fg1_rnl_b ,label=f"{self.fg1}",c='b',lw=2)
        ax2.loglog(fg2_rnl_b ,label=f"{self.fg2}",c='magenta',lw=2)
        ax2.legend(fontsize=20)
        ax2.set_xlabel("$\ell$",fontsize=25)
        ax2.set_ylim(1e-7,1e-2)
        ax2.set_xlim(2,600)

        if save:
            plt.savefig('plots/simFG2.pdf',bbox_inches='tight',dpi=300)
        






class recStat:
    """
    Statistic class for the Reconstruction publication

    Parameters
    ----------
    rec_nofg : object : Reconstruction object with no foregrounds
    rec_fg1 : object : Reconstruction object with foregrounds
    """

    def __init__(self,rec_nofg=None,rec_fg1=None,rec_fg2=None,fg1='s0d0',fg2='s1d1'):
        self.rec_nofg = rec_nofg
        self.rec_fg1 = rec_fg1
        self.rec_fg2 = rec_fg2
        self.fg1 = fg1
        self.fg2 = fg2

    def plot_fg_impact(self,save=False):
        """
        Plot the impact of foregrounds on the reconstruction
        """

        fname = '../Data/paper/recFG.pkl'
        if os.path.isfile(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            rec_nofg = self.rec_nofg
            rec_fg1 = self.rec_fg1
            rec_fg2 = self.rec_fg2
            data = {}
            data['nofg_cl'] = rec_nofg.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
            data['fg2_cl'] = rec_fg2.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
            data['fid'] = rec_fg1.bin_cell(rec_fg1.cl_pp*rec_fg1.Lfac)
            data['fidm'] = rec_fg1.cl_pp*rec_fg1.Lfac
            data['NOFG-MCN0'] = rec_nofg.Lfac*(rec_nofg.MCN0()/rec_nofg.response_mean()**2 )
            data['fg1-MCN0'] = rec_fg1.Lfac*(rec_fg1.MCN0()/rec_fg1.response_mean()**2 )
            data['fg2-MCN0'] = rec_fg2.Lfac*(rec_fg2.MCN0()/rec_fg2.response_mean()**2 )
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
        plt.setp(axs, xlim=(2,620))
        fig.subplots_adjust(hspace=0)


        axs[0].semilogy(data['fidm'],label='Signal',c='grey',lw=2)
        axs[0].semilogy(data['NOFG-MCN0'],label='NOFG $N_L^{(0),MC}$',c='b',ls='-.')
        axs[0].semilogy(data['fg1-MCN0'],label=f"{self.fg1}"+ " $N_L^{(0),MC}$",c='r',ls='-.')
        axs[0].semilogy(data['fg2-MCN0'],label=f"{self.fg2}"+ " $N_L^{(0),MC}$",ls='-.')
        axs[0].semilogy(data['NOFG-MCER']+(data['fidm']/100),label='NOFG $C_L^{MC}$',ls=':',lw=3)
        axs[0].semilogy(data['fg2-MCER']+(data['fidm']/100),label=f"{self.fg2}"+ " $C_L^{MC}$",ls=':',lw=3)
        axs[0].semilogy(data['NOFG-MF'],label='NOFG $C_L^{MF}$',c='y')
        axs[0].semilogy(data['fg2-MF'],label=f"{self.fg2}"+ " $C_L^{MF}$",c='g')
        axs[0].legend(ncol=3, fontsize=16,frameon=False)
        axs[0].set_ylim(1e-9,1e-5)
        axs[0].set_ylabel('$\\frac{L^2 (L + 1)^2}{2\pi} C_L^{\phi\phi}$',fontsize=25)

        axs[1].errorbar(data['B'],fg2_cl.mean(axis=0)/fid,yerr=fg2_cl.std(axis=0)/fid,label=f'{self.fg2}',c='r',fmt='o')
        axs[1].errorbar(data['B'],nofg_cl.mean(axis=0)/fid,yerr=nofg_cl.std(axis=0)/fid,label='NOFG',c='b',fmt='o')
        axs[1].set_ylim(-0.1,2.3)
        axs[1].legend(ncol=2, fontsize=15,loc='upper left')
        axs[1].axhline(1,c='k')
        axs[1].set_ylabel('$\\frac{C_L^{\phi\phi,rec}}{C_L^{\phi\phi,signal}}$',fontsize=25)
        axs[1].set_xlabel("$L$",fontsize=25)
        if save:
            plt.savefig('plots/recFG.pdf', bbox_inches='tight',dpi=300)


    def plot_map_dif(self,idx=35,save=False):
        """
        Plot the difference between the input and reconstructed maps
        """
        fnamei = '../Data/paper/phi_input_35.fits'
        fnameo = '../Data/paper/phi_output_35.fits'

        if os.path.exists(fnamei):
            input = hp.read_map(fnamei)
            print('Input Map Loaded from file')
        else:
            rec_fg2 = self.rec_fg2
            dir_ = "/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CMB_Lensed_Maps/MASS"
            fname = os.path.join(dir_,f"phi_sims_{idx:04d}.fits")
            phi = hp.read_alm(fname)
            DfacL = np.arange(1025)
            Dfac = np.sqrt(DfacL*(DfacL+1))
            Dfac[:10] = 0
            dphi = hp.almxfl(phi,Dfac)
            input = hp.ma(hp.alm2map(dphi,512))
            input.mask = np.logical_not(rec_fg2.mask)
            hp.write_map(fnamei,input)
            print('Input Map Saved to file')
        
        if os.path.exists(fnameo):
            output = hp.read_map(fnameo)
            print('Output Map Loaded from file')
        else:
            output = hp.ma(rec_fg2.deflection_map(idx))
            output.mask = np.logical_not(rec_fg2.mask)
            hp.write_map(fnameo,output)
            print('Output Map Saved to file')

        plt.figure(figsize=(20,20))
        plt.subplots_adjust(wspace=5)
        hp.mollview(input,norm='hist',min=-.0005,max=.001,title='$\\sqrt{L(L+1)}\phi^{input}_{LM}$',sub=(1,2,1),notext=True)
        hp.mollview(output,norm='hist',min=-.0005,max=.001,title='$\\sqrt{L(L+1)}\phi^{WF,rec}_{LM}$',sub=(1,2,2),notext=True)
        if save:
            plt.savefig('plots/recMaps.pdf', bbox_inches='tight',dpi=300)

    def SNR_impact(self):
        """
        Difference in SNR between the foreground and no foreground case
        """
        fname = '../Data/paper/snr.pkl'
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('SNR Loaded from file')
        else:
            data = {}
            rec_nofg = self.rec_nofg
            rec_fg1 = self.rec_fg1
            rec_fg2 = self.rec_fg2
            data['NOFG-SNR'] = rec_nofg.SNR_phi(rdn0=True)
            data['fg1-SNR'] = rec_fg1.SNR_phi(rdn0=True)
            data['fg2-SNR'] = rec_fg2.SNR_phi(rdn0=True)
            pl.dump(data,open(fname,'wb'))
            print('SNR Saved to file')

        SNR_nofg =  data['NOFG-SNR']
        SNR_fg1 = data['fg1-SNR']
        SNR_fg2 = data['fg2-SNR']

        print(f"SNR NOFG: {SNR_nofg:.2f}")
        print(f"SNR FG1: {SNR_fg1:.2f} decreased by {(1-SNR_fg1/SNR_nofg)*100:.2f} %")
        print(f"SNR FG2: {SNR_fg2:.2f} decreased by {(1-SNR_fg2/SNR_nofg)*100:.2f} %")
    
    def plot_qcl_stat(self,save=False):
        fname = '../Data/paper/recQCL.pkl'
        if os.path.exists(fname):
            data = pl.load(open(fname,'rb'))
            print('Data Loaded from file')
        else:
            data = {}
            data['stat']= self.rec_fg2.get_qcl_wR_stat(n=400,n1=True,rdn0=True)
            data['fid'] = self.rec_fg2.cl_pp*self.rec_fg2.Lfac
            data['mcn0'] = self.rec_fg2.Lfac*(self.rec_fg2.MCN0()/self.rec_fg2.response_mean()**2 )
            data['mcn1'] = self.rec_fg2.Lfac*self.rec_fg2.N1
            data['mf'] = self.rec_fg2.Lfac*self.rec_fg2.mean_field_cl()
            data['B'] = self.rec_fg2.B
            pl.dump(data,open(fname,'wb'))
            print('Data Saved to file')

        
        stat = data['stat']
        plt.figure(figsize=(8,7))
        plt.loglog(data['fid'],label='Signal',c='grey',lw=2)
        plt.loglog(data['mcn0'],label='$N_L^{(0)}$',c='r')
        plt.loglog(data['mcn1'],label='$N_L^{(1)}$',c='g')
        plt.loglog(data['mf'],label='$C_L^{MF}$',c='b')
        plt.errorbar(data['B'],stat.mean(axis=0),yerr=stat.std(axis=0),fmt='o',c='k',ms=6,capsize=2,label='Reconstructed')
        plt.xlim(2,600)
        plt.legend(ncol=2, fontsize=20)
        plt.xlabel('$L$',fontsize=25)
        plt.ylabel('$\\frac{L^2 (L + 1)^2}{2\pi} C_L^{\phi\phi}$',fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if save:
            plt.savefig('plots/recQCL.pdf', bbox_inches='tight',dpi=300)
    
    def plot_bin_corr_comp(self,which=1,save=False):
        fname = f'../Data/paper/bin_corr.pkl'
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

        f,(ax1,ax2, axcb) = plt.subplots(1,3, 
            gridspec_kw={'width_ratios':[1,1,0.08]},figsize=(10,5))

        ax1.get_shared_y_axes().join(ax1,ax2)
        g1 = sns.heatmap(data[which]['mcn0'],cmap="coolwarm",cbar=False,ax=ax1)
        g1.set_ylabel('b')
        g1.set_xlabel('b')
        g1.set_title('$N_L^{(0),MC}$')
        g2 = sns.heatmap(data[which]['rdn0'],cmap="coolwarm",ax=ax2, cbar_ax=axcb)
        g2.set_ylabel('b')
        g2.set_xlabel('b')
        g2.set_yticks([])
        g2.set_title('$N_L^{(0),RD}$')
        if save:
            plt.savefig(f'plots/recCor{which}.pdf', bbox_inches='tight',dpi=300)
 
        
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
