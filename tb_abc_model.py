#Tight-binding model for ABC stacked graphene

import numpy as np
from numba.experimental import jitclass # for the class structure
from numba import njit,prange # for extra functions
from numba import int32,float32,complex64
import time
import matplotlib.pyplot as plt

#Establishes constants
pi = np.pi
s2 = np.sqrt(2)
s3 = np.sqrt(3)
G = 4*pi/s3 # [1/a]

spec = [
    ('num_layers',int32),
    ('valley',int32)
]

@jitclass(spec)
class ABC_graphene(object):
    """
    A class to create a tight-binding model for ABC stacked graphene about the K(K') point

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
        
    def create_H(self,kpi,onsite):
        """
        Creates our lower diagonal Hamiltonian matrix given our kpoint and external potential

        Parameters:
            kpi (complex): kpoint w.r.t. valley: kpi = xi*kx + i*ky
                THIS MUST BE IN UNITS OF [k/G]
            onsite (array): onsite energies (manipulate to take into account external electric fields) [meV]

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

        # values of u depend on external field, and are defined/calculated elsewhere
        nu = s3*g/2

        kpi *= G # [k] for standard input

        H = np.zeros((2*self.num_layers,2*self.num_layers),dtype=complex64)

        # Fills in the rest of the Hamiltonian
        for rind in range(0,2*self.num_layers):
            for cind in range(rind,2*self.num_layers): #defines upper triangular

                # main diagonal
                if cind==rind:
                    H[rind][cind] = onsite[int(rind/2)]
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

    def generate_cut(self,num_k,krange,onsite):
        """
        Creates a cut of our energy landscape 

        Parameters:
            num_k (int): dimensionality of k-point partition
            krange (float): how far along our cut to analyze (k/G) 
            onsite (list): if len(list)=1, will treat this as Uext [meV] and not account for screening.
                           if len(list)>1, will treat this as the onsite energies [meV]
        
        Returns:
            energy_array (float): all energy eigenvalues along our cut
        """

        kx_vals = np.linspace(krange,-krange,num_k)
        ky_vals = np.linspace(0,0,num_k)

        if len(onsite)==1: # if not taking into account screening
            u = np.linspace(-onsite[0]/2,onsite[0]/2,self.num_layers)
        else: # if taking into account screening
            if len(onsite) != self.num_layers:
                raise ValueError('Dimension of onsite vector != 1 or number of layers')
            u = np.array(onsite)

        # energy[band][k]
        energy = np.zeros((2*self.num_layers,num_k))
        for k in range(num_k):

            # Defines kpoints
            kx = kx_vals[k] # [k/G]
            ky = ky_vals[k] # [k/G]

            #Creates hamiltonian
            kpi = self.valley*kx + 1j*ky # pi = xi*kx + iky from literature
            H = self.create_H(kpi,u)

            eig = np.linalg.eigvalsh(H) # lower diagonal
            for band,en in enumerate(eig):
                energy[band][k] = en

        return energy

    def generate_grid(self,num_k,krange,onsite,output_vecs):
        """
        Creates a two-dimensional energy grid of the conduction band about our valley

        Parameters:
            num_k (int): dimensionality of k-point partition
            krange (float): analyze square grid of (-krange,krange) (units of k/G)
            onsite (list): if len(onsite)=1, will treat this as Uext [meV] and not account for screening.
                           if len(onsite)>1, will treat this as the onsite energies [meV]
            output_vecs (boolean): whether to output eigenvectors or not

        Returns:
            energy_mesh (nbands*num_k*num_k): if output_all is True
            energy_mesh,vecs: 
        """

        # Creates k-space mesh
        kx_vals = np.linspace(-krange,krange,num_k)
        ky_vals = np.linspace(-krange,krange,num_k)

        if len(onsite)==1: # if not taking into account screening
            u = np.linspace(-onsite[0]/2,onsite[0]/2,self.num_layers)
        else: # if taking into account screening
            if len(onsite) != self.num_layers:
                raise ValueError('Dimension of onsite vector != 1 or number of layers')
            u = np.array(onsite)

        # [band][ky][kx]
        energy_mesh = np.zeros((2*self.num_layers,num_k,num_k))#,dtype=float32)

        # [ky][kx][band][coeff]
        energy_vecs = np.zeros((num_k,num_k,2*self.num_layers,2*self.num_layers),dtype=complex64)

        for kx_ind,kx in enumerate(kx_vals):
            for ky_ind,ky in enumerate(ky_vals):

                # Creates Hamiltonian
                kpi = self.valley*kx + 1j*ky # pi = xi*kx + iky from literature
                H = self.create_H(kpi,u)

                # assumes bottom half of bands occupied and top half of bands are filled
                
                if output_vecs is False: # only stores eigenvalues
                    eig = np.linalg.eigvalsh(H) # lower diagonal
                else: # store eigenvalues and eigenvectors
                    eig,vec = np.linalg.eigh(H) # lower diagonal

                for band,val in enumerate(eig):
                    energy_mesh[band][ky_ind][kx_ind] = val
                
                if output_vecs is True:
                    for band in range(2*self.num_layers):
                        energy_vecs[ky_ind][kx_ind][band] = vec[:,band]

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

    def pot2el(self,delta): 
        """
        Converts 'potential difference between outermost layers' to electric field (= D/epsilon_0) between two parallel plates.
        This effecively goes between the theoretical and experimental data.

        Parameters: 
            delta (float): the potential difference between outermost layers [meV]

        Returns:
            el (float): the electric field [V/nm]
        """

        interlayer_distance = 0.335 # [nm] https://doi.org/10.1103/PhysRevB.82.035409
        hBN_const = (3.29+3.76)/2   # hBN dielectric constant - https://doi.org/10.1038/s41699-018-0050-x 
        graphite_const = 8 # middle-ground of bulk graphite and bilayer graphene dielectric constant https://doi.org/10.1039/C8NA00350E 

        # For a constant electric field (assumed here), potential difference is
        # delta = 1000* (num_layers-1)*(interlayer_distance)/dielectric * E

        return delta/1e3/(self.num_layers-1)/interlayer_distance*graphite_const
    
    def disp2pot(self,el):
        """
        Converts electric field (= D/epsilon_0) between two parallel plates to 'potential difference between outermost layers'
        This effecively goes between the experimental and theoretical data.

        Parameters:
            el (float): the displacement field [V/nm]

        Returns: 
            delta (float): the potential difference between outermost layers [meV]
        """

        interlayer_distance = 0.335 # [nm] https://doi.org/10.1103/PhysRevB.82.035409
        hBN_const = (3.29+3.76)/2   # hBN dielectric constant - https://doi.org/10.1038/s41699-018-0050-x 
        graphite_const = 8 # middle-ground of bulk graphite and bilayer graphene dielectric constant https://doi.org/10.1039/C8NA00350E 

        # For a constant applied electric field (assumed here), potential difference is
        # delta = 1000* (num_layers-1)*(interlayer_distance)/dielectric * E

        return 1e3*el*(self.num_layers-1)*interlayer_distance/graphite_const
        

@njit
def generate_dos(en,Emin,Emax,dE=0.01):
    """
    Generates the density of states (DOS) for our electronic bandstructure

    Parameters:
        model (class): this is a class as defined by ABC_graphene(num_layers,valley)
        en (array): energy array from standard ouput of self.generate_grid (one band).
        Emin (float): minimum energy to analyze
        Emax (float): maximum energy to analyze
        dE (float): width of energy bin

    Returns:
        en_out (array,float): energy axis for our dos
        dos (array,float): density of electronic states (per spin)
    """

    num_bins = round((Emax-Emin)/dE)
    en_out = np.zeros(num_bins)
    dos = np.zeros(num_bins)

    if en.ndim == 2: # self.generate_grid (just one band)

        en_flat = en.flatten()

        for bin in prange(num_bins):
            E0 = Emin + bin*dE
            E1 = Emin + (bin+1)*dE
            en_out[bin] = (E0+E1)/2
            for energy_ind in range(len(en_flat)):
                energy_val = en_flat[energy_ind]
                if E0 <= energy_val and energy_val < E1:
                    dos[bin] += 1

    else:
        raise ValueError('Dimension of en is not 2')

    return en_out , dos/dE # number of counts per energy bin

@njit
def generate_pdos(model,en,vecs,Emin,Emax,dE=0.01,carr=False):
    """
    Generates the partial density of states (PDOS) for our electronic bandstructure

    Parameters:
        model (class): this is a class as defined by ABC_graphene(num_layers,valley)
        en (array): energy array from one energy band of standard ouput of self.generate_grid
        vecs (array): eigenvectors from one energy band standard output of self.generate_grid
        Emin (float): minimum energy to analyze [same unit as en]
        Emax (float): maximum energy to analyze [same unit as en]
        dE (float): width of energy bin [same unit as en]
        carr (boolean): whether to insert a -1/2N factor on each |coefficient|^2 term.  This is required for carrier densities
                        as it compares current system to a system of isolated (non-interacting) graphene layers. The factor of
                        1/2 is from putting this factor on EACH sublattice, of which there are two (A and B).

    Returns:
        en_out (array,float): energy axis for our dos
        pdos (array,float): partial density of electronic states (per spin)
    """

    # finds band minimum/maximum within cutoff momentum (circle)
    k_radius = en.shape[0]/2
    band_minimum,band_maximum = Emax,Emin # initial guess, will be overwritten in this block (switched max-min so first step in loop declares)
    for ky in range(en.shape[0]):
        for kx in range(en.shape[1]):
            if (kx-k_radius)**2 + (ky-k_radius)**2 <= k_radius**2: # if we are within cutoff momentum
                if en[ky][kx] < band_minimum:
                    band_minimum = en[ky][kx]
                if en[ky][kx] > band_maximum:
                    band_maximum = en[ky][kx]

    Emin = max(band_minimum,Emin) # only analyzes the band, even if declared Emin goes below band minimum
    Emax = min(band_maximum,Emax) # only analyzes the band, even if declared Emax goes above band maximum

    for ky in range(en.shape[0]):
        for kx in range(en.shape[1]):
            if (kx-k_radius)**2 + (ky-k_radius)**2 > k_radius**2: # if we are outside cutoff momentum
                en[ky][kx] = band_minimum - 1 # ensures later summations exclude momentums outside the cutoff

    num_bins = round((Emax-Emin)/dE)
    en_out = np.zeros(num_bins)
    pdos = np.zeros((vecs.shape[2],num_bins)) # [coeff][bin]

    if en.ndim == 2: # self.generate_grid (just one band)
        en_flat = en.flatten() # C order flattens all indices (ky,kx)
        vec_flat = vecs.reshape(-1, vecs.shape[-1]) # C order flattens first all but last index (ky,kx)

        if carr is True: # if we are using PDOS for carrier density calculation
            one_over_two_N = 1/2/model.num_layers # compares current system to that of non-interacting layers
        else: # if we are NOT using PDOS for carrier density calculation
            one_over_two_N = 0 # solely adds |coefficient|^2

        for bin in prange(num_bins):
            E0 = Emin + bin*dE
            E1 = Emin + (bin+1)*dE
            en_out[bin] = (E0+E1)/2
            for energy_ind in range(len(en_flat)):
                energy_val = en_flat[energy_ind]
                if E0 <= energy_val and energy_val < E1:
                    for coeff_ind,coeff in enumerate(vec_flat[energy_ind]):
                        pdos[coeff_ind][bin] += np.abs(coeff)**2 - one_over_two_N

    return en_out , pdos/dE # number of counts per energy bin

@njit
def carr_density(krange,en,en_dos,dos):
    """
    This calculates the charge carrier density landscape bound by the fermi momentum (kx,ky)
    in the low temperature limit (where the Fermi-dirac distribution f(E) -> 1).

    n = (2pi)^-2 * (area per kpoint) * int_{E_min}^{E_F} g(E') dE

    Parameters:
        krange (float): analyzing k-space from -krange to krange
        en (array,float): standard output for the energy from self.generate_grid(one band)
        en_dos (array,float): standard output for the energy from generate_dos() or generate_pdos()
        dos (array,float): standard output for the dos from generate_dos() or generate_pdos()

    Returns:
        n (float): electron carrier density [10^12 cm^-2]
    """

    dE = en_dos[1]-en_dos[0] # uniform sampling here, so index 1 and 0 are arbitrary; we just need index n+1 and n

    n = 0.0
    for dos_val in dos: # proper upper/lower integration bounds are already established when calculating en_dos
        n += dos_val*dE

    tot_pts = en.shape[0]*en.shape[1]
    a = 0.246*100 # 10^-9 cm ; lattice constant
    area_per_pt = (2*krange * 4*pi/s3/a)**2 / tot_pts * 1e6 # 10^12 cm^-2

    # Now multiplies number of k-points by area per k-point
    n *= area_per_pt

    # Multiplies factors from original integration over (kx,ky)

    return n/(2*pi)**2

@njit
def carr_density_contour(krange,en,en_dos,dos):
    """
    This calculates the charge carrier density landscape bound by the fermi momentum (kx,ky)
    in the low temperature limit (where the Fermi-dirac distribution f(E) -> 1).

    n = (2pi)^-2 * (area per kpoint) *  int_{E_c}^E(kx,ky) g(E') dE

    Parameters:
        krange (float): analyzing k-space from -krange to krange [k/G]
        en (array,float): standard output for the energy from self.generate_grid(conduction band)
        en_dos (array,float): standard output for the energy from self.generate_dos()
        dos (array,float): standard output for the dos from self.generate_dos()

    Returns:
        n (array,float): a grid storing the carrier density.  Same dimensions as en  [10^12 cm^-2]
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
    area_per_pt = (2*krange * 4*pi/s3/a)**2/tot_pts * 1e6 # 10^12 cm^-2

    # Now multiplies number of k-points by area per k-point
    n *= area_per_pt

    # Multiplies factors from original integration over (kx,ky)

    return n/(2*pi)**2

def hartree_screening(model,num_k,dE,n_top=None,n_bottom=None,U_ext=None,doping=None,onsite=[],mix=0.5,maxiter=100,conv_crit=1e-4,rounding=4):
    """
    Determines onsite energies taking into account screening.  Implimented from:
    - PHYSICAL REVIEW B 81, 125304 (2010)
    - PHYSICAL REVIEW B 80, 195401 (2009)

    Either (n_top and n_bottom) OR (U_ext and doping) must be declared.  They are different approaches to accomplish the same goal: applying some 
    external electric field with some amount of doping.

    Parameters:
        model (class): this is a class as defined by ABC_graphene(num_layers,valley)
        num_k (int): dimensionality of k-point partition
        dE (float): energy bin width for PDOS calculation
        n_top (float): electron density of top gate [x10^12 cm^-2]
        n_bottom (float): electron density of bottom gate [x10^12 cm^-2]
        U_ext (float): potential difference between outermost layers (equivalent to applying an electric field) [meV]
        doping (float): net electron density [x10^12 cm^-2].  Determines the upper limit on our carrier density integrals
        onsite (list): initial guess of onsite energies [meV].  If not declared, will go from -U_ext/2 -> +U_ext/2
        mix (float): mixing weight, the percentage of the NEW step to include; 0 <= mix < = 1; x1 = (1-mix)*x0 + mix*x1
        maxiter (int): maximum number of iterations
        conv_crit (float): convergence criteria for self-consistent process
        rounding (int): number of digits to round resultant onsite energies to
     
    Returns:
        onsite (array): onsite energies [meV]
    """

    # Checking for proper user inputs

    if (n_top is not None and n_bottom is not None): # if declaring n_top and n_bottom
        if (U_ext is not None or doping is not None): # user also declares either U_ext or doping
            raise ValueError('Must declare only (n_top , n_bottom) or (U_ext , doping)')
    if (U_ext is not None and doping is not None): # if declaring U_ext and doping
        if (n_top is not None and n_bottom is not None): # user also declares n_top or n_bottom
            raise ValueError('Must declare only (n_top , n_bottom) or (U_ext , doping)')
    if (n_top is not None and n_bottom is None) or (n_top is None and n_bottom is not None): # if only declaring n_top or only declaring n_bottom
        raise ValueError('Must declare both n_top and n_bottom')
    if (U_ext is not None and doping is None) or (U_ext is None and doping is not None): # if only declaring U_ext or only declaring doping
        raise ValueError('Must declare both U_ext and doping')

    if (mix < 0) or (mix > 1):
        raise ValueError('mixing weight must satisfy: 0 <= mix <= 1')

    if (conv_crit < 10**(-rounding)):
        raise ValueError('Convergence criterion is more accurate than the rounding allows')
    
    if len(onsite) not in [0,model.num_layers]:
        raise ValueError('Length of input onsite energies must be zero (empty list) or equal to the number of layers')

    print('------------------------------------------------------------------',flush=True)
    print('Starting screening calculation')

    num_layers = model.num_layers # number of layers
    k_cutoff = 1/2/pi * (0.502/3.16) # [k/G] cutoff momentum such that furthest points out on our circle satisfy k = gamma1/nu0
    interlayer_distance = 0.335 # [nm] https://doi.org/10.1103/PhysRevB.82.035409
    e_over_2e0 = 1.6021766/2/8.8541878 # electron-charge/2/e0 ; x10^-12 V/nm (cm^2)
    kappa_interlayer = 1 # permittivity of space between layers ; 1 for vacuum, 2.3 for system on SiO2 ; PHYSICAL REVIEW B 80, 195401 (2009)
    buffer = 0.01 # seconds to wait after printing update

    if (n_top is not None and n_bottom is not None): # if the user is declaring the top/bottom gate electron densities
        E_applied = e_over_2e0 * (n_top - n_bottom) # [V/nm]
        U_ext = 1e3 * (num_layers-1)*interlayer_distance*E_applied # [meV]  

        doping = n_top + n_bottom
    
    onsite_ext = list(np.linspace(-U_ext/2,U_ext/2,num_layers)) # onsite energy contribution from external electric field [meV]

    # Initial guess for onsite energies if not declared by user
    if len(onsite) == 0:
        onsite = list(np.linspace(-U_ext/2,U_ext/2,num_layers)) # list for proper input data-type for generate_grid ; will be overwritten below ; [meV]

    converged = False # establishes convergence logic
    for iter in range(maxiter):
        print(f'    Step {iter}:',flush=True)
        print(f'        Onsite: {onsite} meV',flush=True)

        # Calculates energy grid
        print('        Calculating eigensystem',flush=True)
        en_grid,en_vecs = model.generate_grid(num_k,k_cutoff,onsite,True)

        # Determines Fermi energy (chemical potential) to match input doping
        print('        Determining Fermi level',flush=True)

        if (abs(doping) < 1e-4): # if no doping

            extra_band = 0 # number of extra bands to integrate for this doping
            fermi_energy = 0.0 # [meV] # keeps a fermi energy within the energy gap (or at the Dirac point if no applied external field)

        elif (doping > 0): # electron-doped

            extra_band = 1 # number of extra bands to integrate for this doping (the lowest conduction band)

            # vecs = np.zeros((num_k,num_k,2*num_layers),dtype=complex) # [ky][kx][coeff] for lowest conduction band
            # for kx in range(num_k):
            #     for ky in range(num_k):
            #         vecs[ky][kx] = en_vecs[ky][kx][num_layers]
            
            band_min = np.min(en_grid[num_layers]) # conduction band minimum
            energy_difference = np.min(en_grid[num_layers+1]) - band_min # difference between second lowest conduction band min. and lowest conduction band min.

            conduction_electron_density_last = 0 # conduction electron density of the prior step.  This initializes this for later checking
            fermi_energy = band_min+dE # declares the start of our search
            for fermi_energy_index in range(int(energy_difference/dE)): # iterates from lowest cond. band min. to second lowest cond. band min.

                # We put +dE twice for the first step to avoid errors when integrating over one bin
                fermi_energy += dE # maximum on carrier density integrals (max energy to consider for PDOS)

                conduction_dos_en,conduction_dos = generate_dos(en=en_grid[num_layers],Emin=band_min,Emax=fermi_energy,dE=dE)
                conduction_electron_density_current = carr_density(krange=k_cutoff,en=en_grid[num_layers],en_dos=conduction_dos_en,dos=conduction_dos)

                if conduction_electron_density_current > doping: # this process is slowly works up the lowest conduction band until we hit the declared value

                    # Now determines if current or prior step is closest to the user-defined doping.
                    if (abs(conduction_electron_density_current-doping) < abs(conduction_electron_density_last-doping)): # if current step is closer than prior step
                        conduction_electron_density = conduction_electron_density_current
                    else: # if prior step is closer than current step
                        conduction_electron_density = conduction_electron_density_last

                    print(f'            Fermi level: {np.round(fermi_energy,rounding)} meV',flush=True)
                    print(f'            Electron doping: {np.round(conduction_electron_density,rounding)} x10^12 cm^-2',flush=True)
                    break

                conduction_electron_density_last = conduction_electron_density_current

                if (fermi_energy_index == int(energy_difference/dE)-1):
                    raise ValueError('The user declared amount of doping cannot be achieved with solely the lowest conduction band')

        elif (doping < 0): # hole-doped
            
            extra_band = 0 # number of extra bands to integrate for this doping (the highest valence band)

            # vecs = np.zeros((num_k,num_k,2*num_layers),dtype=complex) # [ky][kx][coeff] for lowest conduction band
            # for kx in range(num_k):
            #     for ky in range(num_k):
            #         vecs[ky][kx] = en_vecs[ky][kx][num_layers-1]
            
            band_max = np.max(en_grid[num_layers-1]) # valence band maximum
            energy_difference = band_max - np.max(en_grid[num_layers-2]) # difference between highest valence band max. and second highest valence band max.

            valence_electron_density_last = 0 # valence electron density of the prior step.  This initializes this for later checking
            fermi_energy = band_max-dE # declares the start of our search
            for fermi_energy_index in range(int(energy_difference/dE)): # iterates from lowest cond. band min. to second lowest cond. band min.

                # We put -dE twice for the first step to avoid errors when integrating over one bin
                fermi_energy -= dE # maximum on carrier density integrals (max energy to consider for PDOS)

                valence_dos_en,valence_dos = generate_dos(en=en_grid[num_layers-1],Emin=fermi_energy,Emax=band_max,dE=dE)
                valence_electron_density_current = -1*carr_density(krange=k_cutoff,en=en_grid[num_layers-1],en_dos=valence_dos_en,dos=valence_dos)
                # -1 on valence_electron_density since we are integrating a valence band.  Determines how much below the band gap to set the Fermi level

                if valence_electron_density_current < doping: # this process is slowly works up the lowest conduction band until we hit the declared value

                    # Now determines if current or prior step is closest to the user-defined doping.
                    if (abs(valence_electron_density_current-doping) < abs(valence_electron_density_last-doping)): # if current step is closer than prior step
                        valence_electron_density = valence_electron_density_current
                    else: # if prior step is closer than current step
                        valence_electron_density = valence_electron_density_last

                    print(f'            Fermi level: {np.round(fermi_energy,rounding)} meV',flush=True)
                    print(f'            Electron doping: {np.round(valence_electron_density,rounding)} x10^12 cm^-2',flush=True)
                    break

                valence_electron_density_last = valence_electron_density_current

                if (fermi_energy_index == int(energy_difference/dE)-1):
                    raise ValueError('The user declared amount of doping cannot be achieved with solely the highest valence band')

        else:
            raise ValueError('Something has gone wrong with the doping declaration')

        # Calculates PDOS for bands
        prog_bar = '-' * (num_layers+extra_band) # progress bar
        print(f'        Calculating partial density of states: [{prog_bar}] 0.0%',end="\r",flush=True)
        time.sleep(buffer)

        en_pdos = list() # [band][energy values]
        pdos = list()    # [band][coeff][pdos values]
        for band in range(num_layers+extra_band): # iterates over all valence bands + bands required for doping
            vecs = np.zeros((num_k,num_k,2*num_layers),dtype=complex) # [ky][kx][coeff] for current band
            for kx in range(num_k):
                for ky in range(num_k):
                    vecs[ky][kx] = en_vecs[ky][kx][band]

            band_square_min = np.min(en_grid[band]) # minimum of the band on a square grid.
                                                    # The pdos function will then find it for the inscribed circle (max momentum cutoff)
            en_pdos_append,pdos_append = generate_pdos(model,en=en_grid[band],vecs=vecs,dE=dE,Emin=band_square_min,Emax=fermi_energy,carr=True)

            # Stores energies/pdos for each band
            en_pdos.append(en_pdos_append)
            pdos.append(pdos_append)

            prog_bar = 'X'*(band+1) + '-' * (num_layers+extra_band-(band+1)) # progress bar
            print(f'        Calculating partial density of states: [{prog_bar}] {np.round((band+1)/(num_layers+extra_band)*100,2)}%',end="\r",flush=True)
            time.sleep(buffer)

        print('') # terminal goes to next line

        # Calculates carrier density for each band in each layer

        prog_bar = '-' * (num_layers+extra_band) # progress bar
        print(f'        Calculating carrier densities: [{prog_bar}] 0.0%',end="\r",flush=True)
        time.sleep(buffer)

        electron_density_components = np.zeros((num_layers+extra_band,num_layers)) # [band][layer]
        for band in range(num_layers+extra_band): # iterates over all valence bands + bands required for doping
            for layer in range(num_layers):

                # sublattice A
                electron_density_A = carr_density(krange=k_cutoff,en=en_grid[band],en_dos=en_pdos[band],dos=pdos[band][2*layer]) # [x10^12 cm^-2]

                # sublattice B
                electron_density_B = carr_density(krange=k_cutoff,en=en_grid[band],en_dos=en_pdos[band],dos=pdos[band][2*layer+1]) # [x10^12 cm^-2]

                # combining results from both sublattices
                electron_density_components[band][layer] = electron_density_A + electron_density_B # [x10^12 cm^-2]

            prog_bar = 'X' * (band+1) + '-' * (num_layers+extra_band-band-1)
            print(f'        Calculating carrier densities: [{prog_bar}] {np.round((band+1)/(num_layers+extra_band)*100,2)}%',end="\r",flush=True)
            time.sleep(buffer)

        print('') # terminal goes to next line

        # This block uses electron densities in all layers to determine screened electric field.

        # Finds electron density for each layer (sums over all required bands per layer)
        electron_density = list() # [layer]
        for layer in range(num_layers): # iterates over all layers
            electron_density.append(np.sum(electron_density_components[:,layer]))

        print(f'        electron_density = {list(np.round(electron_density,rounding))} x10^12 cm^-2')
        # NOTE: the net electron density (from this array) will always be zero by construction.  This stems from the |A|^2 + |B|^2 - 1/N within our PDOS.
        # From our normalized eigenvector, it summing the contributions from all layers will be 1 - N*(1/N) = 0.  I will ask Allan about this at our next meeting.

        # Calculates new (screening) electric field between layers
        E_induced = np.zeros(num_layers-1) # will store new electric fields between layers from electrons

        for space_ind in range(num_layers-1): # sums over space between layers

            E_induced[space_ind] -= sum(electron_density[:space_ind+1]) # subtracts densities below space (since electrons pull electric field down)
            E_induced[space_ind] += sum(electron_density[space_ind+1:]) # adds densities above space (since electrons pull electric field up)
                
        E_induced *= e_over_2e0 # [V/nm]   
        E_induced /= kappa_interlayer # accounts for permittivity between layers 

        # print(f'        Applied electric field: {Uext/( 1e3*(num_layers-1)*interlayer_distance )} V/nm' , flush=True)
        # print(f'        (Induced) Electric field: {E_induced} V/nm' , flush=True)

        # Calculates new onsite terms from calculated carrier densities
        onsite_new = list() # places zero voltage at halfway between outermost layers
        if (num_layers%2 == 0): # for an even number of layers ; odd number of spaces
            
            middle_space_index = num_layers//2-1 # for middle space

            V_upper = +E_induced[middle_space_index]*interlayer_distance/2
            onsite_new.append(1000*V_upper) # 1000 * [V] * 1[e] = 1000*[eV] = [meV] ; upper-half

            V_lower = -E_induced[middle_space_index]*interlayer_distance/2
            onsite_new.insert(0,1000*V_lower) # lower-half

            # remaining spaces
            for space_index in range(1,(E_induced.shape[0]-1)//2+1):
                V_upper += E_induced[middle_space_index + space_index]*interlayer_distance
                onsite_new.append(1000*V_upper) # upper-half

                V_lower -= E_induced[middle_space_index - space_index]*interlayer_distance
                onsite_new.insert(0,1000*V_lower) # lower-half

        else: # for an odd number of layers ; even number of spaces

            onsite_new.append(0.0) # for central layer
            V_upper = 0.0 # declaring for later for-loop
            V_lower = 0.0 # declaring for later for-loop

            middle_upper_space_index = E_induced.shape[0]//2 # for upper space of the two center-most spaces
            middle_lower_space_index = E_induced.shape[0]//2-1 # for lower space of the two center-most spaces

            # remaining spaces

            for space_index in range(E_induced.shape[0]//2):
                V_upper += E_induced[middle_upper_space_index + space_index]*interlayer_distance # [eV]
                onsite_new.append(1000*V_upper) # [meV] upper-half

                V_lower -= E_induced[middle_lower_space_index - space_index]*interlayer_distance # [eV]
                onsite_new.insert(0,1000*V_lower) # [meV] lower-half

        # Combines onsite energies from external field and induced fields
        for ind,external_energy in enumerate(onsite_ext):
            onsite_new[ind] += external_energy

        # This now mixes result from prior step and this step.  It is standard for iterative processes, and helps avoid overshooting the true result
        for ind in range(num_layers):
            onsite_new[ind] = (1-mix)*onsite[ind] + mix*onsite_new[ind]

        # Stores array where each element is the difference between new and prior onsite energies (for convergence criterion)
        diff = np.array(onsite_new)-np.array(onsite)

        if np.max(abs(diff)) <= conv_crit: # if maximum difference is less than the convergence criterion
            converged = True
            break

        # Sets up next step
        for onsite_ind,onsite_val in enumerate(onsite_new): # for-loop avoids pointer issue when copying lists
            onsite[onsite_ind] = np.round(onsite_val,rounding)

    if converged is True:
        print('')
        print('Hartree process converged',flush=True)
        print('Self Consistent Electric fields:',flush=True)
        print(f'    Ext = {np.round(U_ext/(num_layers-1)/1000/interlayer_distance,6)} V/nm')
        print(f'    Induced = {np.round(E_induced,rounding)} V/nm') # only taking outermost layers into account
    else:
        print('')
        print('Hartree process did not converge',flush=True)

    return onsite

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
