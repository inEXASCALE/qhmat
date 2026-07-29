function out = gemm_smoothquant_A_pertensor_B_perchannel(A, B, num_bits, alpha, eps, clamp_scale)
    % SmoothQuant + Strategy 1.
    if nargin < 3, num_bits = 8; end
    if nargin < 4, alpha = 0.5; end
    if nargin < 5, eps = 1e-12; end
    if nargin < 6, clamp_scale = []; end
    A = single(A); B = single(B);
    check_gemm_shapes(A, B);

    C_ref = A * B;
    [A_s, B_s, smooth_scale, smooth_info] = smoothquant_transform(A, B, alpha, eps, clamp_scale);

    out = gemm_symmetric_A_pertensor_B_perchannel(A_s, B_s, num_bits, eps);
    metrics = error_metrics(out.C, C_ref, eps);

    out.C_ref = C_ref;
    out.relative_error = metrics.fro_relative_error;
    out.absolute_error = metrics.fro_absolute_error;
    out.error_metrics = metrics;
    out.smoothquant = struct( ...
        'enabled', true, ...
        'alpha', double(alpha), ...
        'smooth_scale', smooth_scale, ...
        'info', smooth_info, ...
        'transform', 'A_smooth = A / s, B_smooth = B * s');
end