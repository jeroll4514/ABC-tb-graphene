#Tight-binding model for ABC stacked graphene

import numpy as np

#Establishes constants
pi = np.pi
s3 = np.sqrt(3)
G = 4*pi/s3 # [1/a]

class ABC_graphene(object):
    """
    A class to create a tight-binding model for
    ABC stacked graphene about the K(K') point

    Parameters:
        num_layers (int): number of layers (>=3)
        valley (int): 1 = K-valley ; -1 = K'-valley
    """

    def __init__(self,num_layers=3,valley=1):
        #Sets user inputs
        self.num_layers = num_layers
        self.valley = valley
    
        if num_layers < 3:
            raise ValueError('Error: num_layers must be >= 3')
        if valley not in [0,1]:
            raise ValueError('Error: user-defined valley must be 0,1')
        
    def create_H(self,kpi,Uext=0):
        """
        Creates our Hamiltonian matrix given our kpoint
        and external potential

        Parameters:
            kpi (complex): kpoint w.r.t. valley: kpi = xi*kx + i*ky
                THIS MUST BE IN UNITS OF [k/G]
            Uext (float): external potential between outermost layers

        Returns:
            H (array,complex): Hamiltonian matrix
        """

        # Defines tight-binding parameters
        # Table 1 in PHYSICAL REVIEW B 82, 035409 (2010)
        delta = -0.0014 # delta  [eV]
        g = np.array([3.16,0.502,-0.0171,-0.377,-0.099,0,0]) # gamma values [eV]
        # values of u depend on external field, and are defined later
        nu = s3*g/2

        kpi *= G # [k] for standard input

        # Puts external potential onto onsite-energies
        u = np.linspace(-Uext/2,Uext/2,self.num_layers)

        H = np.zeros((2*self.num_layers,2*self.num_layers),dtype=complex)

        # Fills in the rest of the Hamiltonian
        for rind in range(0,2*self.num_layers):
            for cind in range(rind,2*self.num_layers): #defines upper triangular

                # main diagonal
                if cind==rind:
                    H[rind][cind] = u[np.floor(rind/2).astype(int)]
                # 1st off diagonal
                if cind==(rind+1):
                    if rind%2==0:
                        H[rind][cind] = nu[0]*np.conj(kpi)
                    else:
                        H[rind][cind] = g[1]
                # 2nd off diagonal
                if cind==(rind+2):
                    H[rind][cind] = nu[4]*np.conj(kpi)
                # 3rd off diagonal
                if cind==(rind+3):
                    if rind%2==0:
                        H[rind][cind] = nu[3]*kpi
                    else:
                        H[rind][cind] = nu[5]*np.conj(kpi)
                # 4th off diagonal
                if cind==(rind+4):
                    H[rind][cind] = nu[6]*kpi
                # 5th off diagonal
                if cind==(rind+5):
                    if rind%2==0:
                        H[rind][cind] = g[2]/2

        # Accounts for higher-energy lattice sites
        H[0][0] += delta
        H[-1][-1] += delta

        return H
    
    def generate_cut(self,num_k=50,krange=0.01,Uext=0):
        """
        Creates a cut of our energy landscape 

        Parameters:
            num_k (int): dimensionality of k-point partition
            krange (float): how far along our cut to analyze (k/G) 
            Uext (float): external potential between outermost layers
        
        Returns:
            energy_array (float): all energy eigenvalues along our cut
        """

        kx_vals = np.linspace(krange,-krange,num_k)
        ky_vals = np.linspace(0,0,num_k)

        energy = list([] for _ in range(2*self.num_layers))
        for ii in range(num_k):

            # Defines kpoints
            kx = kx_vals[ii] # k/G
            ky = ky_vals[ii] # k/G

            #Creates hamiltonian
            kpi = self.valley*kx + 1j*ky # pi = xi*kx + iky from literature
            H = self.create_H(kpi,Uext=Uext)

            eig = np.linalg.eigvalsh(H,UPLO='U')
            for jj,en in enumerate(eig):
                energy[jj].append(en)

        return energy

    def generate_grid(self,num_k=50,kx_min=-0.01,kx_max=0.01,
                      ky_min=-0.01,ky_max=0.01,Uext=0):
        """
        Creates a 2x2 energy grid of the conduction band about our valley

        Parameters:
            num_k (int): dimensionality of k-point partition
            kx_min (float): minimum kx value to analyze (units of k/G)
            kx_max (float): maximum kx value to analyze (units of k/G)
            ky_min (float): minimum ky value to analyze (units of k/G)
            ky_max (float): maximum ky value to analyze (units of k/G)
            Uext (float): external potential between outermost layers

        Returns:
            energy_mesh (array,real): energy landscape of lowest conduction band
        """

        # Creates k-space mesh
        kx_vals = np.linspace(kx_min,kx_max,num_k)
        ky_vals = np.linspace(ky_min,ky_max,num_k)

        energy_mesh = np.zeros((num_k,num_k),dtype=float)
        for ii,kx in enumerate(kx_vals):
            for jj,ky in enumerate(ky_vals):

                # Creates Hamiltonian
                kpi = self.valley*kx + 1j*ky # pi = xi*kx + iky from literature
                H = self.create_H(kpi,Uext=Uext)

                #assumes bottom half of bands occupied and top half of bands are filled
                eig = np.linalg.eigvalsh(H,UPLO='U')
                energy_mesh[jj][ii] = eig[self.num_layers] 

        return energy_mesh
    
    def get_bandgap(self,energy,gamma=False):
        """
        This obtains the bandgap given the total energy of all bands
        (Given standard output of self.generate_cut with np.linalg.eigvalsh)

        Parameters:
            energy (array,float): the output of generate_cut
            gamma (logical): whether to look at just Gamma point (k/G=0) or not

        Returns:
            bandgap (float): the bandgap for our system
        """
        if gamma is False:
            return min(energy[self.num_layers]) - max(energy[self.num_layers-1])
        else:
            k_gamma = round(len(energy[0])/2) #index for gamma point
            return energy[self.num_layers][k_gamma] - energy[self.num_layers-1][k_gamma]
        
    def generate_dos(self,en=[],dE=0.01,Emin=-0.5,Emax=0.5):
        """
        Generates the density of states (DOS) for our electronic bandstructure

        Parameters:
            en (array): energy array from standard ouput of self.generate_cut (if not given, will be calculated)
            dE (float): width of energy bin
            Emin (float): minimum energy to analyze
            Emax (float): maximum energy to analyze

        Returns:
            dos (array,float): density of electronic states
        """

        if en==[]:
            en = self.generate_cut(num_k=1000,krange=0.025)

        num_bins = round((Emax-Emin)/dE)
        dos = np.zeros(num_bins)
        for ii in range(num_bins):
            E0 = Emin + ii*dE
            E1 = Emin + (ii+1)*dE
            for band in en:
                for energy_val in band:
                    if E0 <= energy_val and energy_val < E1:
                        dos[ii] += 1

        return dos/dE #normalizes the number of counts
