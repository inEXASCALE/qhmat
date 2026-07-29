classdef Hmat_node < handle
    properties
        boxNumber         % integer
        parentNumber      % integer
        childrenNumbers   % array of int indices
        
        interactionList   % list of interacting box indices
        neighborNumbers   % list of neighbor box indices
        
        isVisited         % Logical array
        isVisited_nbd     % Logical array
        isLRuseful        % Check if LR useful or not
        
        center            % coordinates (1 x d_dim vector)
        
        charges           % column vector
        potential         % column vector
        
        chargeLocations   % indices of particles belonging to this box

        sq_norm_block_mat
        
        % Low-rank factors
        U 
        S
        V  
        % Dense matrices
        dense_nbd_Matrix
        dense_self_Matrix
    end
    
    methods
        function obj = Hmat_node(d_dim)
            % Constructor
            obj.boxNumber = -1;
            obj.parentNumber = -1;
            
            obj.childrenNumbers = zeros(1, 2^d_dim, 'int32');
            
            obj.interactionList = int32([]);
            obj.neighborNumbers = int32([]);
            
            obj.isVisited      = sparse([], [], [], 100000, 1);
            
            obj.center = zeros(1, d_dim);
            
            obj.charges = [];
            obj.potential = [];
            
            obj.chargeLocations = int32([]);
            
            obj.U = {};
            obj.S = {};
            obj.V = {};
        end
    end
end
