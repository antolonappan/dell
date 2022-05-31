import curvedsky as cs
import mpi



class Delens:

    def __init__(self,lib_dir,recon):
        self.lib_dir = lib_dir
        self.recon = recon
        
        if mpi.rank == 0:
            os.makedirs(self.lib_dir, exist_ok=True)
        
    def get_cinv_Emode(self,idx):
        return self.recon.get_falm_sim(idx,ret='E')
    
    def get_reconst_qlm(self):
        pass
    
    def get_wiener_filt(self):
        pass
    
    def get_b_template(self):
        pass
    
    def get_input_Bmode(self):
        pass
    
    def get_TempXinput(self):
        pass
    
    def get_delensed_bmode(self):
        pass