# qhmat

## Description of the parameters in the [main_file.m](Hmat/main_file.m)
1) d_dim = The dimension of the space. The balanced 2^d_dim-tree will be generated accordingly. For example, if d_dim=3, the resulting tree will be an oct-tree.
1) nParticlesInLeafAlong1D = The maximum number of particles along 1D at the leaf level. The maximum number of particles at the leaf level N_max = nParticlesInLeafAlong1D^d_dim (this decides the depth of the tree $L$).  
3) LR_epsilon (ε) = Target accuracy in the low-rank compression.
4) kernel_choice = Choice of your desired kernel function (see kernel_menu.md file and select the number from there).
5) hodlr_level = This decides the switching level $\ell$ (where the admissibility conditions will be switched) in our hybrid admissibility condition. For example, if hodlr_level = 1, then switching level $\ell = L-1$.
6) eta = The parameter used in the standard admissibility condition. A different choice of eta will generate a different $\mathcal{H}$-matrix.
7) is_sym = The kernel matrix is symmetric or not (true/false).
8) u = Working precision for mixed precision H-matrix representation storage. 
9) u_mvp = Working precision for matrix-vector product with the resulting H-matrix representation.
10) The code can be run in mixed precision as well as uniform precision settings (more info can be found in the main_file.m).
11) There is a parameter N_max in [createHmat.m](Hmat/createHmat.m) responsible for building the entire matrix to check the global error caused by the adaptive mixed precision hybrid hierarchical representation. So make sure that you have enough memory to build the matrix.


## One example
Set the input parameters as follows in the [main_file.m](Hmat/main_file.m)

```
d_dim = 3;
d_dimRootN = [20];
nParticlesInLeafAlong1D = 4;
LR_epsilon = [1e-2];
kernel_choice = 3;
hodlr_level = 1;
eta = sqrt(d_dim)/2.0;  
particle_location_choice = "uniform";
is_sym = true;
