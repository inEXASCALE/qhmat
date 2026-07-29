function [Q, params] = quantize_symmetric_per_channel(X, axis, num_bits, eps)
    % Per-channel symmetric quantization:  X ~= scale(channel) * Q.
    % axis is 0-based (Python convention).
    if nargin < 3, num_bits = 8; end
    if nargin < 4, eps = 1e-12; end

    X = single(X);
    if axis < 0, axis = axis + ndims(X); end
    if axis < 0 || axis >= ndims(X)
        error('quantize_symmetric_per_channel:axis', ...
              'Invalid axis=%d for X.ndim=%d.', axis, ndims(X));
    end
    dim = axis + 1;

    [qmin, qmax] = get_symmetric_quant_range(num_bits);
    qdtype = get_integer_dtype(num_bits);

    reduceDims = setdiff(1:ndims(X), dim);
    max_abs = abs(X);
    for d = reshape(reduceDims, 1, [])
        max_abs = max(max_abs, [], d);
    end
    max_abs = squeeze(max_abs);
    scale = max_abs / qmax;
    scale = single(max(scale, eps));
    zero_point = zeros(size(scale), 'int64');

    sh = ones(1, ndims(X));
    sh(dim) = size(X, dim);
    scale_bc = reshape(scale, sh);

    Q = npround(X ./ scale_bc);
    Q = cast(max(min(Q, qmax), qmin), qdtype);

    params = make_params(scale, zero_point, qmin, qmax, axis, qdtype);
end