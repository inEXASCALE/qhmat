function [groups, totalMemory_bits, precisionMap] = adaptiveBit_LRfactors(U, S, V, normA, tol, bitWidths)

%--------------------------------------------------------------------------
% Adaptive mixed-bitwidth SVD factorization using a max-norm error bound.
%
% T = U*S*V'
%
% Balanced representation:
%
%       T = U0*V0'
%
% where
%
%       U0 = U*sqrt(S)
%       V0 = V*sqrt(S)
%
% The approximation is represented as
%
%       T_hat = sum_k groups(k).U * groups(k).V'
%
% where each channel can have a different bit width.
%
% Error bound:
%
%       ||T - T_hat||_max
%           <= sum_k c(b_k) ||Sigma_k||_*
%
% where
%
%       c(b) = 1/q_b + 1/(4q_b^2)
%       q_b = 2^(b-1)-1
%
% Small singular values are assigned to lower precision first.
%
% INPUT
%   U         : m x r left singular vectors
%   S         : r x r diagonal singular-value matrix
%   V         : n x r right singular vectors
%   normA     : global norm used for error budget
%   tol       : relative tolerance
%   bitWidths : available bit widths, e.g.
%               [2 4 6 8 10 12 14 16 32]
%
% OUTPUT
%   groups(k).indices
%   groups(k).bits
%   groups(k).U       = quantized U*sqrt(Sigma)
%   groups(k).V       = quantized V*sqrt(Sigma)
%   groups(k).sigmaSum
%   groups(k).errorBound
%   groups(k).memoryU
%   groups(k).memoryV
%   groups(k).memoryTotal
%
%   precisionMap : bit width assigned to each singular component
%   totalMemory_bits : total storage in bits
%--------------------------------------------------------------------------

    %% Singular values

    sigma = diag(S);
    r = length(sigma);
    
    %% Sort available bit widths
    
    bitWidths = sort(bitWidths,'ascend');
    
    nPrec = length(bitWidths);
    
    %% Global error budget
    
    budget = tol * normA;
    
    %% Quantization constants
    
    q = 2.^(bitWidths - 1) - 1;
    

    DelU = max(abs(U), [], 'all')./(2*q);
    DelV = max(abs(V), [], 'all')./(2*q);

    %c_old = 1./q + 1./(4*q.^2);
    
    c = DelU + DelV + DelU .* DelV;
    
    %% ---------------------------------------------------------------
    % Balanced low-rank factors
    %
    %       U0 = U*sqrt(Sigma)
    %       V0 = V*sqrt(Sigma)
    %
    % so that
    %
    %       T = U0*V0'
    % ---------------------------------------------------------------
    
    sqrtSigma = diag(sqrt(sigma));
    
    U0 = U * sqrtSigma;
    V0 = V * sqrtSigma;
    
    %% Initialize groups
    
    groups = struct( ...
        'indices',{}, ...
        'bits',{}, ...
        'U',{}, ...
        'V',{}, ...
        'sigmaSum',{}, ...
        'errorBound',{}, ...
        'memoryU',{}, ...
        'memoryV',{}, ...
        'memoryTotal',{});
    
    precisionMap = zeros(1,r);
    
    remaining = r;
    groupID = 1;
    
    %% ---------------------------------------------------------------
    % Start with LOWEST precision.
    %
    % bitWidths is ascending:
    %
    %       [2 4 6 8 10 12 14 16 32]
    %
    % Therefore start with 2-bit precision.
    %
    % Singular values are ordered:
    %
    %       sigma(1) >= sigma(2) >= ... >= sigma(r)
    %
    % Therefore start with the smallest singular values.
    % ---------------------------------------------------------------
    
    for p = 1:nPrec
    
        if remaining == 0
            break;
        end
    
        b  = bitWidths(p);
        cb = c(p);
    
        idx = remaining;
    
        group = [];
        sigmaSum = 0;
    
        %% Add smallest singular values
    
        while idx >= 1
    
            trialSigmaSum = sigmaSum + sigma(idx);
    
            trialError = cb * trialSigmaSum;
    
            if trialError <= budget
    
                group = [idx, group];
    
                sigmaSum = trialSigmaSum;
    
                idx = idx - 1;
    
            else
    
                break;
    
            end
    
        end
    
        %% No singular values fit this precision
    
        if isempty(group)
            continue;
        end
    
        %% Error contribution
    
        groupError = cb * sigmaSum;
    
        %% -----------------------------------------------------------
        % Extract BALANCED factors
        %
        %       Uk = U(:,group)*sqrt(Sigma(group))
        %       Vk = V(:,group)*sqrt(Sigma(group))
        %
        % Hence
        %
        %       Uk*Vk' = U(:,group)*Sigma(group)*V(:,group)'
        % -----------------------------------------------------------
    
    
        %% Store channel
        groups(groupID).indices = group;
        groups(groupID).bits = b;
        groups(groupID).U = U0(:,group);
        groups(groupID).V = V0(:,group);
        groups(groupID).sigmaSum = sigmaSum;
        groups(groupID).errorBound = groupError;
    
        %% Memory
        groups(groupID).memoryU = numel(groups(groupID).U) * b;
        groups(groupID).memoryV = numel(groups(groupID).V) * b;
        groups(groupID).memoryTotal = groups(groupID).memoryU + groups(groupID).memoryV;
    
        %% Precision map
        precisionMap(group) = b;
    
        %% Update error budget
        budget = budget - groupError;
    
        %% Update remaining singular components
        remaining = min(group) - 1;
        groupID = groupID + 1;
    end
    
    %% ---------------------------------------------------------------
    % Remaining dominant singular values
    %
    % Assign them to the highest available precision.
    % ---------------------------------------------------------------
    
    if remaining > 0
        p = nPrec;
        b  = bitWidths(p);
        cb = c(p);
        group = 1:remaining;
        sigmaSum = sum(sigma(group));
        groupError = cb * sigmaSum;
    
    
    
        %% Store
        groups(groupID).indices = group;
        groups(groupID).bits = b;
        groups(groupID).U = U0(:,group);
        groups(groupID).V = V0(:,group);
        groups(groupID).sigmaSum = sigmaSum;
        groups(groupID).errorBound = groupError;
    
        %% Memory
        groups(groupID).memoryU = numel(groups(groupID).U) * b;
        groups(groupID).memoryV = numel(groups(groupID).V) * b;
        groups(groupID).memoryTotal = groups(groupID).memoryU + groups(groupID).memoryV;
    
        %% Precision map
        precisionMap(group) = b;
    
    end
    
    %% Total memory
    
    totalMemory_bits = 0;
    
    for k = 1:numel(groups)
        totalMemory_bits = totalMemory_bits + groups(k).memoryTotal;
    end
end