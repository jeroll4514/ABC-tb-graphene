# ABC-tb-graphene
This contains code to implement a tight-binding model for ABC stacked (rhombohedral) graphene.  It is generalized to n layers, where it incorporates interactions 3-layers-deep.  Specifically, this implements the model from: PHYSICAL REVIEW B 82, 035409 (2010).  It uses numba for efficiency and includes several functions:
+ create_H: calculates the Hamiltonian matrix at a given k-point about the K(K') valley.
+ generate_cut: calculates the energy eigenvalues along ky=0 through the K(K') valley
+ generate_grid: calculates the energy landscape on a square grid about the K(K') valley.  Will also return the eigenvectors for all points if needed.
+ get_bandgap: determines the true bandgap or explicitly the energy gap at our high-symmetry K(K') point.
+ find_flat_U: finds the potential difference between the outermost layers, yielding the flattest minimum conduction band.
+ generate_dos: calculates the density-of-states of our energy landscape (typically just the minimum conduction band)
+ carr_density: calculates the (electron) charge carrier density landscape.  This has the upper limit of integration as E(kx,ky) instead of the standard E_fermi.
+ TRS_check: checks if an input system is time-reversal symmetric

To use this class, one must simply import it at the beginning of one's code.  So if the class file (tb_ABC_model.py) is within the running directory, this is:

import tb_ABC_model as tb

Then one can use the class/functions with the prefactor tb.XXX

Example: 

graphene_4L = tb.ABC_graphene(num_layers=3,valley=1)
