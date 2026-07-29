function Xhat = dequantize_symmetric(Q, params)
    % Symmetric dequantization:  X_hat = scale * Q.
    Qf = single(Q);
    if isempty(params.axis)
        Xhat = single(params.scale) .* Qf;
        return;
    end
    sh = ones(1, ndims(Q));
    sh(params.axis + 1) = size(Q, params.axis + 1);
    scale_bc = reshape(single(params.scale), sh);
    Xhat = scale_bc .* Qf;
end