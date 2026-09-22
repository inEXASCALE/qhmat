function out = gemm_symmetric_A_perchannel_B_perchannel(A, B, num_bits, eps)
    % Strategy 2: A per-row, B per-column symmetric quantization.
    %   Acc = Q_A @ Q_B ;  C_hat(m,n) = Acc(m,n) * s_A(m) * s_B(n)
    if nargin < 3, num_bits = 8; end
    if nargin < 4, eps = 1e-12; end
    A = single(A); B = single(B);
    check_gemm_shapes(A, B);

    K = size(A, 2);
    C_ref = A * B;

    [Q_A, p_A] = quantize_symmetric_per_channel(A, 0, num_bits, eps);
    [Q_B, p_B] = quantize_symmetric_per_channel(B, 1, num_bits, eps);

    accumulator_dtype = get_accumulator_dtype(p_A.dtype);
    Acc = integer_gemm_accumulate(Q_A, Q_B, accumulator_dtype);

    C_hat = single(double(Acc) .* double(p_A.scale(:)) .* double(p_B.scale(:).'));

    metrics = error_metrics(C_hat, C_ref, eps);

    qA = struct('quantization','per_channel_symmetric','channel_axis',0, ...
                'channel_meaning','rows / tokens');
    qB = struct('quantization','per_channel_symmetric','channel_axis',1, ...
                'channel_meaning','columns / output channels');
    out = build_gemm_out(C_hat, C_ref, metrics, Q_A, Q_B, Acc, K, p_A, p_B, qA, qB);
end