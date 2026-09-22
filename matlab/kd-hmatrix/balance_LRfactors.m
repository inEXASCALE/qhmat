function [U_bal, V_bal] = balance_LRfactors(U, V)
    % U : m x r
    % V : r x n
    
    % Infinity norm (maximum absolute entry) of each column
    u_max = max(abs(U), [], 1);   
    v_max = max(abs(V'), [], 1);    
    
    % Avoid division by zero
    u_max = max(u_max, eps);
 
    % Diagonal entries
    d = sqrt(v_max ./ u_max);
    
    % Diagonal scaling matrix
    D = diag(d);

    U_bal = U * D;
    V_bal = D \ V;
end