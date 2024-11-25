#Tight-binding model for ABC stacked graphene

import numpy as np
from numba.experimental import jitclass # for the class structure
from numba import njit,prange # for extra functions
from numba import int32,float32,complex64

#Establishes constants
pi = np.pi
s3 = np.sqrt(3)
G = 4*pi/s3 # [1/a]

spec = [
    ('num_layers',int32),
    ('valley',int32)
]

@jitclass(spec)
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
        if valley not in [-1,1]:
            raise ValueError('Error: user-defined valley must be -1,1')
        
    def create_H(self,kpi,Uext):
        """
        Creates our lower diagonal Hamiltonian matrix given our kpoint
        and external potential

        Parameters:
            kpi (complex): kpoint w.r.t. valley: kpi = xi*kx + i*ky
                THIS MUST BE IN UNITS OF [k/G]
            Uext (float): external potential between outermost layers [meV]

        Returns:
            H (array,complex): Hamiltonian matrix
        """

        # Defines tight-binding parameters
        # Table 1 in PHYSICAL REVIEW B 82, 035409 (2010)
        delta = -0.0014*1000 # delta  [meV]
        g = np.array([3.16,0.502,-0.0171,-0.377,-0.099,0,0])*1000 # gamma values [meV]

        # Experimental tight-binding parameters
        # Tight-binding model section in arXiv:2408.15233
        # delta = 0 # [eV]
        # g = np.array([3.1,0.38,-0.0083,-0.29,-0.141,0,0])*1000 # gamma values [meV]

        # values of u depend on external field, and are defined later
        nu = s3*g/2

        kpi *= G # [k] for standard input

        # Puts external potential onto onsite-energies
        u = np.linspace(-Uext/2,Uext/2,self.num_layers)

        H = np.zeros((2*self.num_layers,2*self.num_layers),dtype=complex64)

        # Fills in the rest of the Hamiltonian
        for rind in range(0,2*self.num_layers):
            for cind in range(rind,2*self.num_layers): #defines upper triangular

                # main diagonal
                if cind==rind:
                    H[rind][cind] = u[int(rind/2)]
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

        return np.conj(H).T # lower diagonal

    def generate_cut(self,num_k,krange,Uext):
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

        # energy[band][k]
        energy = np.zeros((2*self.num_layers,num_k))
        for k in range(num_k):

            # Defines kpoints
            kx = kx_vals[k] # k/G
            ky = ky_vals[k] # k/G

            #Creates hamiltonian
            kpi = self.valley*kx + 1j*ky # pi = xi*kx + iky from literature
            H = self.create_H(kpi,Uext)

            eig = np.linalg.eigvalsh(H) # lower diagonal
            for band,en in enumerate(eig):
                energy[band][k] = en

        return energy

    def generate_grid(self,num_k,krange,Uext,output_vecs):
        """
        Creates a 2x2 energy grid of the conduction band about our valley

        Parameters:
            num_k (int): dimensionality of k-point partition
            krange (float): analyze square grid of (-krange,krange) (units of k/G)
            Uext (float): external potential between outermost layers
            output_vecs (boolean): whether to output eigenvectors or not

        Returns:
            energy_mesh (nbands*num_k*num_k): if output_all is True
            energy_mesh,vecs: 
        """

        # Creates k-space mesh
        kx_vals = np.linspace(-krange,krange,num_k)
        ky_vals = np.linspace(-krange,krange,num_k)

        energy_mesh = np.zeros((2*self.num_layers,num_k,num_k),dtype=float32)

        # [ky][kx][band][coeff]
        energy_vecs = np.zeros((num_k,num_k,2*self.num_layers,2*self.num_layers),dtype=complex64)

        for ii,kx in enumerate(kx_vals):
            for jj,ky in enumerate(ky_vals):

                # Creates Hamiltonian
                kpi = self.valley*kx + 1j*ky # pi = xi*kx + iky from literature
                H = self.create_H(kpi,Uext)

                # assumes bottom half of bands occupied and top half of bands are filled
                
                if output_vecs is False: # only stores eigenvalues
                    eig = np.linalg.eigvalsh(H) # lower diagonal
                else: # store eigenvalues and eigenvectors
                    eig,vec = np.linalg.eigh(H) # lower diagonal

                for band,val in enumerate(eig):
                    energy_mesh[band][jj][ii] = val
                
                if output_vecs is True:
                    for band in range(2*self.num_layers):
                        energy_vecs[jj][ii][band] = vec[:,band]

        return energy_mesh,energy_vecs
    
    def get_bandgap(self,energy,gamma):
        """
        This obtains the bandgap given the total energy of all bands
        (Given standard output of self.generate_cut with np.linalg.eigvalsh)

        Parameters:
            energy (array,float): the output of generate_cut
            gamma (boolean): whether to look at just Gamma point (k/G=0) or not

        Returns:
            bandgap (float): the bandgap for our system
        """
        if gamma is False:
            return min(energy[self.num_layers]) - max(energy[self.num_layers-1])
        else:
            k_gamma = round(len(energy[0])/2) #index for gamma point
            return energy[self.num_layers][k_gamma] - energy[self.num_layers-1][k_gamma]
    
    def find_flat_U(self,Umin,Umax,Udim,dE):
        """
        Will find the potential difference U which yields the flattest band near our Fermi surface.
        This will return the U which returns the highest DOS and the energy where this occurs.

        Parameters:
            Umin (float): the minimum U to check
            Umax (float): the maximum U to check
            Udim (int): the number of U values to check
            dE (float): the width of our energy bins for the DOS

        Returns:
            Uopt (float): U with flattest band
            en_flat (float): energy where our flattness occurs
        """

        Uvals = np.linspace(Umin,Umax,Udim)
        Uopt = 0 # will store the U value with the flattest band (highest DOS)
        temp_max = 0 # will store maximum DOS
        for U in Uvals:
            # print(f'Testing U_ext = {U} meV',flush=True)
            # print(f'    Generating grid',flush=True)
            en,t = self.generate_grid(500,0.025,U,False)
            en_cond = en[self.num_layers]
            Emin = np.min(en_cond)
            Emax = Emin+3 # +3 here is from seeing our maximum dos typically occuring within 3 meV of minimum
            # print(f'        Min energy: {Emin} meV')
            # print(f'    Generating dos',flush=True)
            en_dos,dos = generate_dos(en=en_cond,dE=dE,Emin=Emin,Emax=Emax) # dos for conduction band
            # print(f'        Max dos: {max(dos)}')
            if max(dos) > temp_max:
                temp_max = max(dos)
                Uopt = U
                en_flat = en_dos[np.argmax(dos)]

        return Uopt,en_flat

    def pot2disp(self,delta): 
        """
        Converts 'potential difference between outermost layers' to displacement field between two parallel plates.
        This effecively goes between the theoretical and experimental data.

        Parameters: 
            delta (float): the potential difference between outermost layers [meV]

        Returns:
            disp (float): the displacement field [e_0^-1 V/nm]
        """

        interlayer_distance = 0.335 # [nm] https://doi.org/10.1103/PhysRevB.82.035409
        hBN_const = (3.29+3.76)/2   # hBN dielectric constant - https://doi.org/10.1038/s41699-018-0050-x 
        graphite_const = 8 # middle-ground of bulk graphite and bilayer graphene dielectric constant https://doi.org/10.1039/C8NA00350E 

        # For a constant electric field (assumed here), potential difference is
        # delta = 1000* (num_layers-1)*(interlayer_distance)/dielectric * E

        return delta/1e3/(self.num_layers-1)/interlayer_distance*graphite_const
    
    def disp2pot(self,disp):
        """
        Converts displacement field between two parallel plates to 'potential difference between outermost layers'
        This effecively goes between the experimental and theoretical data.

        Parameters:
            disp (float): the displacement field [e_0^-1 V/nm]

        Returns: 
            delta (float): the potential difference between outermost layers [meV]
        """

        interlayer_distance = 0.335 # [nm] https://doi.org/10.1103/PhysRevB.82.035409
        hBN_const = (3.29+3.76)/2   # hBN dielectric constant - https://doi.org/10.1038/s41699-018-0050-x 
        graphite_const = 8 # middle-ground of bulk graphite and bilayer graphene dielectric constant https://doi.org/10.1039/C8NA00350E 

        # For a constant applied electric field (assumed here), potential difference is
        # delta = 1000* (num_layers-1)*(interlayer_distance)/dielectric * E

        return 1e3*disp*(self.num_layers-1)*interlayer_distance/graphite_const
        

@njit
def generate_dos(en,dE=0.01,Emin=-500,Emax=500):
    """
    Generates the density of states (DOS) for our electronic bandstructure

    Parameters:
        en (array): energy array from standard ouput of self.generate_grid
        dE (float): width of energy bin
        Emin (float): minimum energy to analyze
        Emax (float): maximum energy to analyze

    Returns:
        en_out (array,float): energy axis for our dos
        dos (array,float): density of electronic states (per spin)
    """

    num_bins = round((Emax-Emin)/dE)
    en_out = np.zeros(num_bins)
    dos = np.zeros(num_bins)

    if en.ndim == 2: # self.generate_grid (just lowest conduction band)
        N_tot = en.shape[0]*en.shape[1] # total number of k-points in discretization
        ax_band = 0
        ax_k = 1
        en_flat = en.flatten()

        for bin in prange(num_bins):
            E0 = Emin + bin*dE
            E1 = Emin + (bin+1)*dE
            en_out[bin] = (E0+E1)/2
            for energy_ind in range(len(en_flat)):
                energy_val = en_flat[energy_ind]
                if E0 <= energy_val and energy_val < E1:
                    dos[bin] += 1
            # for energy_val in en_flat:
            #     if E0 <= energy_val and energy_val < E1:
            #         dos[bin] += 1
            # for k_ind in range(en.shape[ax_k]):
            #     for band in range(en.shape[ax_band]):
            #         energy_val = en[band][k_ind]
            #         if E0 <= energy_val and energy_val < E1:
            #             dos[bin] += 1

    elif en.ndim == 3: # self.generate_grid(all bands)

        raise NotImplementedError('New developments have made this not functional')

        N_tot = en.shape[1]*en.shape[2] # total number of k-points in discretization
        ax_band = 0
        ax_kx = 1
        ax_ky = 2

        for bin in prange(num_bins):
            E0 = Emin + bin*dE
            E1 = Emin + (bin+1)*dE
            en_out[bin] = (E0+E1)/2
            for kx_ind in range(en.shape[ax_kx]):
                for ky_ind in range(en.shape[ax_ky]):
                    for band in range(en.shape[ax_band]):
                        energy_val = en[band][ky_ind][kx_ind]
                        if E0 <= energy_val and energy_val < E1:
                            dos[bin] += 1

    return en_out , dos/dE # number of counts per energy bin

@njit
def carr_density(krange,en,en_dos,dos):
    """
    This calculates the charge carrier density landscape bound by the fermi momentum (kx,ky)
    in the low temperature limit (where the Fermi-dirac distribution f(E) -> 1).

    n = (2pi)^-2 * int_{E_c}^E(kx,ky) g(E') dE

    Parameters:
        krange (float): analyzing k-space from -krange to krange
        en (array,float): standard output for the energy from self.generate_grid(conduction band)
        en_dos (array,float): standard output for the energy from self.generate_dos()
        dos (array,float): standard output for the dos from self.generate_dos()

    Returns:
        n (array,float): a grid storing the carrier density.  Same dimensions as en
    """

    n = np.zeros(en.shape)
    dE = en_dos[1]-en_dos[0]
    Emin = np.min(en) # lower bound of integration of conduction band
    Emin_ind = np.argmin(np.abs(en_dos-Emin)) # finds energy index for lower bound in en_dos

    # Performs our desired integral for all possible limits (all possible chemical potentials) in krange.
    carr_integral = np.zeros(en_dos.shape)
    for Emax_ind in prange(len(en_dos)):

        for dos_val in dos[Emin_ind:Emax_ind+1]:
            # Adds term in our Riemann integral when: Emin <= Eprime <= Emax
            carr_integral[Emax_ind] += dos_val*dE # number of k-points

            # this last block is equivalent (but more efficient) to the following (easier to understand) block
            # for Eprime_ind,Eprime in enumerate(en_dos):
            #     if Emin <= Eprime and Eprime <= Emax:
            #         n[ky][kx] += dos[Eprime_ind]*dE
    
    # Now assigns previously calculated integrals to proper energies in k-grid
    for kx in range(en.shape[1]):
        for ky in range(en.shape[0]):
            Emax = en[ky][kx] # upper limit of integration
            Emax_ind = np.argmin(np.abs(en_dos-Emax)) # finds energy index for current integration upper limit
            n[ky][kx] = carr_integral[Emax_ind]

    tot_pts = en.shape[0]*en.shape[1]
    a = 0.246*100 # 10^-9 cm ; lattice constant
    area_per_pt = (2*krange)**2*(4*pi/s3/a)**2/tot_pts * 1e6 # 10^12 cm^-2

    # Now multiplies number of k-points by area per k-point
    n *= area_per_pt

    # Multiplies factors from original integration over (kx,ky)

    return n/(2*pi)**2

@njit
def find_surface_recombination(grid):
    """
    Finds value where contours of the input grid merge into one surface (instead of many smaller pockets)

    Parameters:
        grid (float,array): for this project will be the std output of generate_grid(one-band-only) or carr_density()

    Returns:
        grid_val_out: this is the value where our contour merges into one surface
    """

    value_found = False # will store if a closed contour exists
    center_ind = int(grid.shape[0]/2) # index going through horizontal center of grid

    for grid_val in np.unique(grid): # iterates over all unique values in grid

        occurances = 0 # number of occurances of a specific number
        
        cut_ind_store = np.zeros(40) # will store index where grid hits value of interest
        for cut_ind in range(grid.shape[1]-1): # looks across grid cut
            grid_center1 = grid[center_ind][cut_ind]
            grid_center2 = grid[center_ind][cut_ind+1]
            if (grid_center1 <= grid_val and grid_val < grid_center2) \
                    or (grid_center2 <= grid_val and grid_val < grid_center1):
                cut_ind_store[occurances] = cut_ind
                occurances += 1
            
            if occurances == 40:
                raise ValueError('Number of surfaces exeeds 20; something is wrong with the grid')

        # Checks if surface intersects with middle cut twice and that both occurances are NOT on
        # the same side (left-right).  If they are on the same side, then we have three isolated pockets
        # due to 3-fold symmetry
        if occurances == 2 and sum(cut_ind_store) < grid.shape[1]:
            value_found = True
            grid_val_out = grid_val
            break

    if value_found is False:
        raise ValueError('The surface with which you seek does not exist.')

    return grid_val_out

def TRS_check(sys1,sys2):
    """
    This is a function to check if two models are related by time-reversal-symmetry (TRS).
    Follows notation shown in Equations 11,12 in arXiv:2406.19348

    Parameters:
        sys1 (class): this is a class as defined by ABC_graphene(num_layers,valley)
        sys2 (class): this is a class as defined by ABC_graphene(num_layers,valley)

    Returns:
        output (boolean): True(False) implies sys1 and sys2 are(aren't) related by TRS.
    """

    en1,vec1 = sys1.generate_grid(output_all=True,output_vecs=True)
    en2,vec2 = sys2.generate_grid(output_all=True,output_vecs=True)

    if en1.shape[0] != en2.shape[0] or en1.shape[1] != en2.shape[1]:
        print(f'Two systems are NOT TRS: have different dimensionality')
        return False

    # First checks the energies are equivalent
    for band in range(en1.shape[0]):
        for kx in range(en1.shape[2]):
            for ky in range(en1.shape[1]):

                # E(+,k) - E(-,-k) 
                diff = en1[band][ky][kx]-en2[band][(en2.shape[1]-1)-ky][(en2.shape[2]-1)-kx]

                if abs(diff) > 1e-8:
                    print(f'Two systems are NOT TRS: Energy difference in band {band} at index ({kx,ky})')
                    return False

    # Second checks if the eigenvectors are equivalent
    for kx in range(en1.shape[2]):
        for ky in range(en1.shape[1]):
            for band in range(en1.shape[0]):
                for coeff in range(vec1.shape[3]):

                    # z(+,l,k) - conj[ z(-,l,-k) ]
                    diff = vec1[ky][kx][band][coeff] - np.conj(vec2[(en2.shape[1]-1)-ky][(en2.shape[2]-1)-kx][band][coeff])

                    if abs(diff) > 1e-8:
                        print(f'Two systems are NOT TRS: Eigenvec. coeff difference in band {band} at index ({kx,ky})')
                        return False

    return True
