classdef HmatKD < handle
    properties
        % Problem data
        N
        d_dim
        particle_loc
        kernelFun

        % H-matrix parameters
        minBlockSize
        eta
        LR_epsilon
        max_level

        % Trees
        clusterRoot
        blockRoot

        % Cluster numbering
        nextBoxNumber = 0

        % Permutation maps
        perm
        iperm

        % Dense reference matrix in permuted ordering
        Afull

        % Max norm of A
        Amax

        % Working precision bit
        FWP = 64

        % H-matrix root
        Hroot

        % Keep different records
        Hstats

        % Bit set
        bit_set
    end

    methods
        %% Constructor
        function obj = HmatKD(particle_loc, minBlockSize, kernelFun, eta, LR_epsilon, bit_set)
            obj.particle_loc = particle_loc;
            obj.N = size(particle_loc,1);
            obj.d_dim = size(particle_loc,2);
            obj.kernelFun = kernelFun;
            obj.minBlockSize = minBlockSize;
            obj.eta = eta;
            obj.LR_epsilon = LR_epsilon;
            obj.max_level = max(1, ceil(log2(max(obj.N / obj.minBlockSize, 1))));
            obj.clusterRoot = [];
            obj.blockRoot = [];
            obj.perm = [];
            obj.iperm = [];
            obj.Afull = [];
            obj.Hroot = [];
            obj.Hstats = struct();
            obj.bit_set = bit_set;
        end

        %% Build cluster tree
        % Recursively splits particle indices into a binary KD-tree.
        % Also builds the leaf permutation perm and inverse permutation iperm.
        function buildClusterTree(obj)
            obj.nextBoxNumber = 0;
            idx = 1:obj.N;
            obj.clusterRoot = obj.buildCluster(idx, -1, 0);
            obj.perm = obj.getLeafOrder(obj.clusterRoot);
            obj.iperm = zeros(obj.N,1);
            obj.iperm(obj.perm) = 1:obj.N;
        end

        %% Recursive KD-tree cluster construction
        % Builds one cluster node, splitting along the coordinate with largest span.
        function node = buildCluster(obj, idx, parent, level)
            node = struct();
            obj.nextBoxNumber = obj.nextBoxNumber + 1;
            node.boxNumber = obj.nextBoxNumber;
            node.parentNumber = parent;
            node.level = level;

            idx = idx(:).';
            node.indices = idx;

            pts = obj.particle_loc(idx,:);
            node.bmin = min(pts, [], 1);
            node.bmax = max(pts, [], 1);
            node.center = 0.5 * (node.bmin + node.bmax);
            node.diam = norm(node.bmax - node.bmin, 2);

            if numel(idx) <= obj.minBlockSize || level >= obj.max_level || numel(idx) <= 1
                node.isLeaf = true;
                node.children = {};
                return;
            end

            span = node.bmax - node.bmin;
            [~, dim] = max(span);
            [~, order] = sort(pts(:,dim));
            idx = idx(order);

            mid = floor(numel(idx)/2);
            if mid < 1 || mid >= numel(idx)
                node.isLeaf = true;
                node.children = {};
                return;
            end

            node.isLeaf = false;
            node.children = cell(1,2);
            node.children{1} = obj.buildCluster(idx(1:mid), node.boxNumber, level+1);
            node.children{2} = obj.buildCluster(idx(mid+1:end), node.boxNumber, level+1);
        end

        %% Extract leaf order
        % Returns the particle indices in left-to-right leaf traversal order.
        function perm = getLeafOrder(obj, node)
            if node.isLeaf
                perm = node.indices;
                return;
            end
            perm = [];
            for k = 1:numel(node.children)
                perm = [perm, obj.getLeafOrder(node.children{k})];
            end
        end

        %% Build dense reference matrix in permuted ordering
        % Afull(i,j) = K(p(i), p(j))
        function buildMatrix(obj)
            N = obj.N;
            p = obj.perm;
            A = zeros(N,N);

            for i = 1:N
                xi = obj.particle_loc(p(i),:);
                for j = 1:N
                    xj = obj.particle_loc(p(j),:);
                    A(i,j) = obj.kernelFun(xi, xj);
                end
            end

            obj.Afull = A;
        end

        %% Cluster diameter
        % Returns the diameter of a cluster bounding box.
        function d = clusterDiameter(~, node)
            d = node.diam;
        end

        %% Distance between two bounding boxes
        % Returns Euclidean distance between two axis-aligned boxes.
        function r = clusterDistance(~, node1, node2)
            dim = numel(node1.bmin);
            gap = zeros(1,dim);

            for k = 1:dim
                if node1.bmax(k) < node2.bmin(k)
                    gap(k) = node2.bmin(k) - node1.bmax(k);
                elseif node2.bmax(k) < node1.bmin(k)
                    gap(k) = node1.bmin(k) - node2.bmax(k);
                else
                    gap(k) = 0;
                end
            end

            r = norm(gap, 2);
        end

        %% Eta admissibility condition
        % A block is admissible if max(diam1, diam2) <= eta * dist.
        function tf = isAdmissible(obj, node1, node2)
            if node1.boxNumber == node2.boxNumber
                tf = false;
                return;
            end

            diam = min(obj.clusterDiameter(node1), obj.clusterDiameter(node2));
            dist = obj.clusterDistance(node1, node2);
            tf = (dist > 0) && (diam <= obj.eta * dist);
        end

        %% Build block tree
        % Recursively partitions the cluster-tree block interaction matrix.
        function buildBlockTree(obj)
            if isempty(obj.clusterRoot)
                obj.buildClusterTree();
            end
            obj.blockRoot = obj.buildBlockNode(obj.clusterRoot, obj.clusterRoot);
        end

        %% Recursive block construction
        % Builds the block cluster tree.
        function node = buildBlockNode(obj, rowNode, colNode)
            node = struct();
            node.row = rowNode;
            node.col = colNode;
            node.admissible = obj.isAdmissible(rowNode, colNode);
            node.isLeaf = rowNode.isLeaf && colNode.isLeaf;
            node.children = {};

            if node.admissible || node.isLeaf
                return;
            end

            nr = numel(rowNode.children);
            nc = numel(colNode.children);
            node.children = cell(nr,nc);

            for i = 1:nr
                for j = 1:nc
                    node.children{i,j} = obj.buildBlockNode(rowNode.children{i}, colNode.children{j});
                end
            end
        end

        %% Build a dense submatrix from original particle indices
        % Used when a block is dense or when compressing an admissible block.
        function A = buildDenseSubmatrix(obj, rowIdx, colIdx)
            nr = numel(rowIdx);
            nc = numel(colIdx);
            A = zeros(nr,nc);

            for i = 1:nr
                xi = obj.particle_loc(rowIdx(i),:);
                for j = 1:nc
                    xj = obj.particle_loc(colIdx(j),:);
                    A(i,j) = obj.kernelFun(xi, xj);
                end
            end
        end



        % Build the full matrix (not necessary!!!!!)
        function getMaxnorm(obj)
            if isempty(obj.Afull)
                obj.buildMatrix();
            end
            obj.Amax = max(abs(obj.Afull), [], 'all');
        end


        %% Build the H-matrix
        % Assembles the recursive block tree into an H-matrix structure.
        function [Hroot, stats] = buildHMatrix(obj)
            if isempty(obj.blockRoot)
                obj.buildBlockTree();
            end
            [Hroot, stats] = obj.assembleBlock(obj.blockRoot);
            obj.Hroot = Hroot;
            obj.Hstats = stats;
        end


        %% Assemble one block recursively
        % Stores row and column indices in permuted coordinates.
        function [B, stats] = assembleBlock(obj, node)
            idxR = node.row.indices;
            idxC = node.col.indices;
            r = obj.iperm(idxR);
            c = obj.iperm(idxC);

            stats.lowRankStorageFWP = 0;
            stats.lowRankStorageQuant = 0;

            stats.denseStorage = 0;

            if node.admissible
                A = obj.buildDenseSubmatrix(idxR, idxC);
                [U, S, V, ~] = LR_compress(A, "svd", obj.LR_epsilon);

                [groups, LocalMem, precisionMap] = adaptiveBit_LRfactors(U, S, V, obj.LR_epsilon, obj.Amax, obj.bit_set);

                %plotBitwidthMap(precisionMap, obj.bit_set);

                % blockSize = size(A);
                % fprintf("Admissible\n");
                % fprintf('LR block: size = %d x %d, numerical rank = %d\n', ...
                %     blockSize(1), blockSize(2), numRank);

                % fprintf('LR block: size = %d x %d, numerical rank = %d\n', ...
                %     blockSize(1), blockSize(2), size(U,2));

                B = struct();
                B.type = 'LR';
                B.admissible = true;
                B.row = r;
                B.col = c;
                B.nrows = numel(r);
                B.ncols = numel(c);
                B.rank = size(U,2);

                B.groups = groups;
                B.LocalMem = LocalMem;
                B.precisionMap = precisionMap;

                stats.lowRankStorageFWP = (numel(U) + numel(V)) * obj.FWP;

                stats.lowRankStorageQuant = LocalMem;
                return;
            end

            if node.row.isLeaf && node.col.isLeaf
                A = obj.buildDenseSubmatrix(idxR, idxC);
                m = size(A,1);
                n = size(A,2);

                B = struct();
                B.row = r;
                B.col = c;
                B.nrows = m;
                B.ncols = n;
                B.admissible = false;

                % Keep diagonal leaf blocks dense
                if node.row.boxNumber == node.col.boxNumber
                    B.type = 'dense';
                    B.A = A;
                    stats.denseStorage = numel(A) * obj.FWP;
                    return;
                end

                % Try low-rank compression for off-diagonal leaf blocks
                [U, S, V, ~] = LR_compress(A, "svd", obj.LR_epsilon);

                [groups, LocalMem, precisionMap] = adaptiveBit_LRfactors(U, S, V, obj.LR_epsilon, obj.Amax, obj.bit_set);

                rr = size(U,2);

                denseCost = m * n * obj.FWP;
                lrCostFWP = (numel(U) + numel(V)) * obj.FWP;

                % Compress only if it saves storage
                if lrCostFWP < denseCost
                    B.type = 'LR';
                    B.rank = rr;

                    B.groups = groups;
                    B.LocalMem = LocalMem;
                    B.precisionMap = precisionMap;

                    stats.lowRankStorageFWP = lrCostFWP;

                    stats.lowRankStorageQuant = LocalMem;
                else
                    B.type = 'dense';
                    B.A = A;

                    stats.denseStorage = denseCost;
                    % fprintf('Kept leaf block %d x %d dense (LR cost %d >= dense cost %d)\n', ...
                    %     m, n, lrCost, denseCost);
                end
                return;
            end

            B = struct();
            B.type = 'node';
            B.admissible = false;
            B.row = r;
            B.col = c;
            B.nrows = numel(r);
            B.ncols = numel(c);
            B.children = cell(size(node.children));

            for i = 1:size(node.children,1)
                for j = 1:size(node.children,2)
                    [B.children{i,j}, childStats] = obj.assembleBlock(node.children{i,j});
                    stats.lowRankStorageFWP = stats.lowRankStorageFWP + childStats.lowRankStorageFWP;

                    stats.lowRankStorageQuant = stats.lowRankStorageQuant + childStats.lowRankStorageQuant;

                    stats.denseStorage = stats.denseStorage + childStats.denseStorage;
                end
            end
        end

        %% H-matrix multiply with a vector or matrix
        % If x is N-by-k, returns y = H*x as N-by-k.
        function y = HmatVec(obj, H, x)
            if isempty(obj.perm)
                obj.buildClusterTree();
            end

            x = obj.ensure2D(x);
            xp = x(obj.perm,:);
            yp = obj.HmatVecNode(H, xp);

            y = zeros(obj.N, size(x,2));
            y(obj.perm,:) = yp;
        end

        %% Recursive H-matrix multiply on permuted data
        % Returns the product in permuted ordering.
        function y = HmatVecNode(obj, H, x)
            switch H.type
                case 'dense'
                    y = zeros(obj.N, size(x,2));
                    y(H.row,:) = H.A * x(H.col,:);

                case 'LR'
                    y = zeros(obj.N, size(x,2));
                    y(H.row,:) = quant_block_mvp(H.groups, x(H.col,:));

                case 'node'
                    y = zeros(obj.N, size(x,2));
                    for i = 1:size(H.children,1)
                        for j = 1:size(H.children,2)
                            child = H.children{i,j};
                            if ~isempty(child)
                                y = y + obj.HmatVecNode(child, x);
                            end
                        end
                    end

                otherwise
                    error('Unknown H-matrix block type');
            end
        end


        %% Reconstruct dense matrix from H-matrix
        % function Arec = reconstructDense(obj, H)
        %     if isempty(obj.perm)
        %         obj.buildClusterTree();
        %     end
        % 
        %     Arec = zeros(obj.N, obj.N);
        %     Arec = obj.fillDenseRecursive(H, Arec);
        % end


        %% Fill dense matrix from H-matrix blocks
        % function A = fillDenseRecursive(obj, H, A)
        %     switch H.type
        %         case 'dense'
        %             A(H.row, H.col) = H.A;
        % 
        %         case 'LR'
        %             A(H.row, H.col) = H.U * H.V;
        % 
        %         case 'node'
        %             for i = 1:size(H.children,1)
        %                 for j = 1:size(H.children,2)
        %                     child = H.children{i,j};
        %                     if ~isempty(child)
        %                         A = obj.fillDenseRecursive(child, A);
        %                     end
        %                 end
        %             end
        % 
        %         otherwise
        %             error('Unknown H-matrix block type');
        %     end
        % end


        %% Reconstruct dense matrix from H-matrix
        function ArecQ = reconstructQuantH(obj, H)
            if isempty(obj.perm)
                obj.buildClusterTree();
            end
            

            ArecQ = zeros(obj.N, obj.N);

            ArecQ = obj.fillDenseRecursiveQuantH(H, ArecQ);
        end


        %% Fill dense matrix from H-matrix blocks
        function A = fillDenseRecursiveQuantH(obj, H, A)
            switch H.type
                case 'dense'
                    A(H.row, H.col) = H.A;

                case 'LR'
                    A(H.row, H.col) = reconstructLRBlock(H);

                case 'node'
                    for i = 1:size(H.children,1)
                        for j = 1:size(H.children,2)
                            child = H.children{i,j};
                            if ~isempty(child)
                                A = obj.fillDenseRecursiveQuantH(child, A);
                            end
                        end
                    end

                otherwise
                    error('Unknown H-matrix block type');
            end
        end


        %% Ensure input is 2D
        % Converts a vector x into an N-by-1 matrix.
        function X = ensure2D(~, X)
            if isvector(X)
                X = X(:);
            end
        end

        %% Test against dense matrix
        % Compares H*(x) or H*(X) to dense multiplication using Afull.
        function [relErr,absErr,yH,yFull] = testHmatVec(obj, H, x)
            if isempty(obj.Afull)
                obj.buildMatrix();
            end
            if nargin < 3 || isempty(x)
                x = randn(obj.N,1);
            end

            x = obj.ensure2D(x);
            xp = x(obj.perm,:);
            yp = obj.Afull * xp;

            yFull = zeros(obj.N, size(x,2));
            yFull(obj.perm,:) = yp;

            yH = obj.HmatVec(H, x);

            absErr = norm(yFull - yH, 'fro');
            relErr = absErr / max(norm(yFull, 'fro'), eps);
        end

        %% Multiple random tests
        % Runs several random H-matrix vector/matrix product tests.
        function report = runAccuracyTest(obj, H, nTests)
            if nargin < 3
                nTests = 5;
            end

            rel = zeros(nTests,1);
            absE = zeros(nTests,1);

            for k = 1:nTests
                x = randn(obj.N,1);
                [rel(k), absE(k)] = obj.testHmatVec(H, x);
            end

            report = struct();
            report.nTests = nTests;
            report.meanRelErr = mean(rel);
            report.maxRelErr = max(rel);
            report.meanAbsErr = mean(absE);
            report.maxAbsErr = max(absE);
        end

        %% H-matrix reconstruction error
        % Computes ||Afull - Arec||_F / ||Afull||_F in permuted ordering.
        % function [relErr, absErr, Arec] = reconstructionError(obj, H)
        %     if isempty(obj.Afull)
        %         obj.buildMatrix();
        %     end
        %     if nargin < 2 || isempty(H)
        %         H = obj.Hroot;
        %     end
        %     if isempty(H)
        %         H = obj.buildHMatrix();
        %     end
        % 
        %     Arec = obj.reconstructDense(H);
        % 
        %     absErr = norm(obj.Afull - Arec, 'fro');
        %     relErr = absErr / max(norm(obj.Afull, 'fro'), eps);
        % end

        %% Quant H-matrix reconstruction error
        % Computes ||Afull - Arec||_F / ||Afull||_F in permuted ordering.
        function [relErr, absErr, Arec] = reconstructionErrorQuantH(obj, H)
            if isempty(obj.Afull)
                obj.buildMatrix();
            end
            if nargin < 2 || isempty(H)
                H = obj.Hroot;
            end
            if isempty(H)
                H = obj.buildHMatrix();
            end

            Arec = obj.reconstructQuantH(H);

            absErr = norm(obj.Afull - Arec, 'fro');
            relErr = absErr / max(norm(obj.Afull, 'fro'), eps);
        end



        %% Get all leaves
        % Returns a cell array of all leaf nodes.
        function leaves = getLeaves(obj, node)
            leaves = [];
            if node.isLeaf
                leaves{1} = node;
                return;
            end

            for k = 1:numel(node.children)
                leaves = [leaves, obj.getLeaves(node.children{k})];
            end
        end

        %% Visualize particle geometry
        % Plots the particle cloud.
        function visualizeGeometry(obj)
            figure;
            if obj.d_dim >= 2
                scatter(obj.particle_loc(:,1), obj.particle_loc(:,2), 20, 'filled');
                xlabel('x_1');
                ylabel('x_2');
            else
                scatter(obj.particle_loc(:,1), zeros(obj.N,1), 20, 'filled');
                xlabel('x');
            end
            axis equal;
            grid on;
            title('Particle locations');
        end

        %% Visualize KD-tree partition
        function plotKDTree(obj)

            if isempty(obj.clusterRoot)
                error('Build cluster tree first using buildClusterTree().');
            end

            figure;
            hold on;
            axis equal;
            box on;

            % Plot particles
            scatter(obj.particle_loc(:,1), ...
                obj.particle_loc(:,2), ...
                30,'filled');

            % Draw KD boxes
            obj.plotKDNode(obj.clusterRoot);

            xlabel('x');
            ylabel('y');
            title('KD-tree partition');
            hold off;

        end


        %% Recursive plotting of KD nodes
        function plotKDNode(obj,node)

            % Only for 2D
            if obj.d_dim ~= 2
                error('Visualization implemented only for 2D.');
            end

            xmin = node.bmin(1);
            xmax = node.bmax(1);
            ymin = node.bmin(2);
            ymax = node.bmax(2);

            rectangle('Position',...
                [xmin,ymin,xmax-xmin,ymax-ymin],...
                'EdgeColor','k');

            % Plot leaf numbers
            if node.isLeaf

                c = node.center;

                text(c(1),c(2),...
                    sprintf('%d',node.boxNumber),...
                    'FontSize',10,...
                    'Color','r');

            else

                for k = 1:numel(node.children)
                    obj.plotKDNode(node.children{k});
                end

            end

        end


        %% Visualize original dense matrix
        % Shows the matrix in input order
        function visualizeOriginal(obj)
        
            N = obj.N;
            Aorig = zeros(N,N);
        
            for i = 1:N
                xi = obj.particle_loc(i,:);
                for j = 1:N
                    xj = obj.particle_loc(j,:);
                    Aorig(i,j)=obj.kernelFun(xi,xj);
                end
            end
        
            figure;
            imagesc(Aorig);
            axis equal tight;
            colorbar;
            title('Original dense matrix');
        
        end        

        %% Visualize permuted dense matrix
        % Shows the reordered dense matrix Afull.
        function visualizePermutationPattern(obj)
            if isempty(obj.Afull)
                obj.buildMatrix();
            end
            figure;
            imagesc(obj.Afull);
            axis equal tight;
            colorbar;
            title('Permuted dense matrix');
        end

        %% Visualize admissibility pattern
        % Displays which leaf blocks are admissible.
        function visualizeAdmissibility(obj)
            if isempty(obj.blockRoot)
                obj.buildBlockTree();
            end

            leaves = obj.getLeaves(obj.clusterRoot);
            m = numel(leaves);
            P = zeros(m,m);

            for i = 1:m
                for j = 1:m
                    P(i,j) = obj.isAdmissible(leaves{i}, leaves{j});
                end
            end

            figure;
            imagesc(P);
            axis equal tight;
            colorbar;
            title('Admissibility pattern');
        end

        function visualizeHMatrixBlocks(obj)
            if isempty(obj.blockRoot)
                obj.buildBlockTree();
            end

            figure;
            hold on;
            obj.drawBlockRecursive(obj.blockRoot);
            axis equal tight;
            set(gca, 'YDir', 'reverse');
            % xlabel('Column index (permuted)');
            % ylabel('Row index (permuted)');
            % title('H-matrix structure');
            hold off;
        end


        function drawBlockRecursive(obj, node)
            rowIdx = node.row.indices;
            colIdx = node.col.indices;

            r = obj.iperm(rowIdx);
            c = obj.iperm(colIdx);

            x1 = min(c) - 0.5;
            x2 = max(c) + 0.5;
            y1 = min(r) - 0.5;
            y2 = max(r) + 0.5;

            if node.admissible
                faceColor = [0.2 0.8 0.2];   % green
                rectangle('Position', [x1 y1 x2-x1 y2-y1], ...
                    'FaceColor', faceColor, ...
                    'EdgeColor', 'k', ...
                    'LineWidth', 0.5);
            elseif node.isLeaf
                faceColor = [0.9 0.2 0.2];   % red
                rectangle('Position', [x1 y1 x2-x1 y2-y1], ...
                    'FaceColor', faceColor, ...
                    'EdgeColor', 'k', ...
                    'LineWidth', 0.5);
            else
                for i = 1:size(node.children,1)
                    for j = 1:size(node.children,2)
                        child = node.children{i,j};
                        if ~isempty(child)
                            obj.drawBlockRecursive(child);
                        end
                    end
                end
            end
        end

    end
end