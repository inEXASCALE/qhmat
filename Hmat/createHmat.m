function createHmat(charges, d_dim, eta, d_dimRootN, hodlr_level, nParticlesInLeafAlong1D, LR_epsilon, kernel_choice, particle_location_choice, is_sym, isNBDcompressed)    
    N = d_dimRootN^d_dim;

    nLevels = ceil(log(d_dimRootN/nParticlesInLeafAlong1D) / log(2));
    L = 1; 

    % Set a value, it will build entire matrix of size N_max X N_max.
    N_max = 10000;
    
    H = HmatTree(d_dim, eta, N, N_max, nLevels, hodlr_level, nParticlesInLeafAlong1D, L, LR_epsilon, kernel_choice, is_sym, isNBDcompressed);
    H.createTree();
    H.assign_Center_Locations();
    H.assign_Tree_Interactions();


    %% %%%%%%%%%%%%%% PARTICLE LOCATION (SELECT ONLY ONE) %%%%%%%%%%%%%%%%%%%%
    H.assign_Particle_Locations(particle_location_choice); % uniform, random 
    %H.assign_Particle_Locations_Surface(particle_location_choice); %2D: circle, star 3D: sphere, polyhedron
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    H.assign_Charge_Locations();
    H.assign_NonLeaf_Charge_Locations();
    H.assign_Charges(charges);
    H.compute_Hmat_maxnorm();
    H.compress_Hmat_Block_Matrices();


    if(isNBDcompressed)
        H.compress_Junction_Block_Matrices(); % Note, compress when the flag is true and hodlr_level=0;
    end
    if(hodlr_level>0)
        H.compress_HODLR_Block_Matrices();
    end
    H.assemble_Dense_Matrices();


    % Matrix-vector product
    H.fast_MVP_Hmat();
    if(isNBDcompressed)
        H.fast_MVP_Junction(); % Joining level
    end
    if(hodlr_level>0)
        H.fast_MVP_HODLR();
    end
    H.dense_MVP_leaf();

    poten = H.collectPotential();

    H.reorder(poten);

    H.get_storage();

    if(~is_sym)
        fprintf("Hmat storage\n");
        disp(H.Hmat_storage);
        fprintf("Joining level storage\n");
        disp(H.nadm_storage);
        fprintf("HODLR storage\n");
        disp(H.HODLR_strorage);
    end

    fileID = fopen('Results/result_QuantH','a+');

    fprintf(fileID,"\n~~~~~~~*************** AMP HMAT ****************~~~~~~~\n");
    fprintf(fileID,"Number of particles: %d\n  ", N);
    fprintf(fileID,"\nParameters:\n");
    fprintf(fileID,"HODLR level-> %d\t  ", hodlr_level);
    fprintf(fileID,"LR Tol.-> %d\t  ", LR_epsilon);
    fprintf(fileID,"Dim-> %d\t  ", d_dim);
    fprintf(fileID,"Min particle at leaf-> %d\t  ", nParticlesInLeafAlong1D^d_dim);
    fprintf(fileID,"Kernel choice-> %d\t  ", kernel_choice);
    fprintf(fileID,"isNBDcompressed-> %d\t  ", isNBDcompressed);
    fprintf(fileID,"isSymmetric-> %d\t  ", is_sym);
    fprintf(fileID,"eta-> %d\n", eta);
    fprintf(fileID,"Particle location-> %s\n", particle_location_choice);
    fprintf(fileID,"=============================\n");

    if(N<=N_max)
        txt = evalc('H.compute_rel_error_mvp()');  
        fprintf(fileID, "%s", txt);
        fprintf(fileID,"=============================\n");
    end

    fprintf(fileID,"Level of the tree: ");
    fprintf(fileID,"%d\n", nLevels);
    fprintf(fileID,"=============================\n");

    fprintf(fileID,"Low-rank Storage in quantized form (GB): ");
    fprintf(fileID,"%d\n", H.LR_storage * 1.25e-10);
    fprintf(fileID,"=============================\n");

    fprintf(fileID,"Dense matrices Storage (GB): ");
    fprintf(fileID,"%d\n", H.Dense_storage * 1.25e-10);
    fprintf(fileID,"=============================\n");

    fprintf(fileID,"Total storage of quantized H-matrix (GB): ");
    fprintf(fileID,"%d\n", H.storage * 1.25e-10);
    fprintf(fileID,"=============================\n");


    fprintf(fileID,"Storage gain (with fp64): ");
    fprintf(fileID,"%d\n", H.storageFP64/H.storage);
    fprintf(fileID,"=============================\n");

    if(N<=N_max)
        txt = evalc('H.compute_Hmat_error()');  % Capture printed output
        fprintf(fileID, "%s\n", txt);
        fprintf(fileID,"=============================\n");
        txt = evalc('H.compute_MVP_error()');  % Backward error in MVP
        fprintf(fileID, "%s", txt);
    end
    fprintf(fileID,"*************** == END OF RESULT == ****************\n");

    fclose(fileID);   % Close the file

    if(d_dim == 2 && N<=8000)
        H.visualize_2d_tree();
    elseif(d_dim == 3 && N<=8000)
        H.visualize_3d_tree();
    end

    % If you want to see the hierarchical matrix
    if(~is_sym && N<=6400)
        plotHmatrix(H);
    end
end

