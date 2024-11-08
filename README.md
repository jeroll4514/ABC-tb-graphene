# ABC-tb-graphene
This contains code to implement a tight-binding model for ABC stacked (rhombohedral) graphene.  It is generalized to n layers, where it incorporates interactions 3-layers-deep.  Specifically, this implements the model from: PHYSICAL REVIEW B 82, 035409 (2010).

To use this class, one must simply import it at the beginning of one's code.  So if the class file (tb_ABC_model.py) is within the running directory, this is:

from tb_ABC_model import ABC_graphene
