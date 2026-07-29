function [Q, params] = quantize_symmetric_per_tensor(X, num_bits, eps)
    % Per-tensor symmetric quantization:  X ~= scale * Q,  zero_point = 0.
    if nargin < 2, num_bits = 8; end
    if nargin < 3, eps = 1e-12; end

    X = single(X);
    [qmin, qmax] = get_symmetric_quant_range(num_bits);
    qdtype = get_integer_dtype(num_bits);

    max_abs = max(abs(X(:)));
    if max_abs < eps
        scale = single(1.0);
    else
        scale = single(max_abs / qmax);
    end
    zero_point = int64(0);

    Q = npround(X / scale);
    Q = cast(max(min(Q, qmax), qmin), qdtype);

    params = make_params( ...
        single(scale), ...
        int64(zero_point), ...
        qmin, qmax, [], qdtype);
end