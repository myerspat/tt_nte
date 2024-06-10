# Pu 239 data
class Inputs_fissile_brick:
    def __init__(
        self,
        Z = 94,
        A = 239,
        nu_fission  = 3.24 * 0.081600,
        scatter_gtg = 0.225216,
        total       = 0.32640,
        Num_ordinates = [2],
        
        # desired solvers
        
        MGES     = True,
        MPower   = True,
        TTPower  = True,
        QTTPower = False,
        
        # Desired Plots
        Eigenvec_err = False,
        Eigenval_err = True,
        compression_ratio = False,
            ):
        
        self.Z = Z
        self.A = A
        self.nu_fission = nu_fission
        self.scatter_gtg = scatter_gtg,
        self.total       = total,
        self.Num_ordinates = Num_ordinates       
        self.MGES = MGES,
        self.MPower = MPower,
        self.TTPower = TTPower,
        self.QTTPower = QTTPower,        
        self.Eigenvec_err = Eigenvec_err,
        self.Eigenval_err = Eigenval_err,
        self.compression_ratio = compression_ratio,