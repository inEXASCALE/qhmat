function [A_smooth, B_smooth, s, info] = smoothquant_transform(A, B, alpha, eps, clamp_scale)
    % SmoothQuant transform for GEMM C = A*B with per-K-channel scale s(k):
    %   A_smooth(:,k) = A(:,k) / s(k)
    %   B_smooth(k,:) = B(k,:) * s(k)
    % so  A*B == A_smooth*B_smooth  up to FP rounding.
    %   s(k) = max_abs_A(k)^alpha / max_abs_B(k)^(1-alpha)
    if nargin < 3, alpha = 0.5; end
    if nargin < 4, eps = 1e-12; end
    if nargin < 5, clamp_scale = []; end

    A = single(A); B = single(B);
    check_gemm_shapes(A, B);

    if alpha < 0 || alpha > 1
        error('smoothquant_transform:alpha', 'alpha must be in [0, 1].');
    end

    a_absmax = max(abs(A), [], 1);   % [1, K]   (Python axis=0)
    b_absmax = max(abs(B), [], 2).'; % [1, K]   (Python axis=1)
    a_absmax = double(max(a_absmax, eps));
    b_absmax = double(max(b_absmax, eps));

    s = (a_absmax .^ alpha) ./ (b_absmax .^ (1.0 - alpha));
    s = max(s, eps);

    if ~isempty(clamp_scale)
        lo = clamp_scale(1); hi = clamp_scale(2);
        if lo <= 0 || hi <= 0 || lo > hi
            error('smoothquant_transform:clamp', ...
                  'clamp_scale must be (lo, hi) with 0 < lo <= hi.');
        end
        s = min(max(s, lo), hi);
    end
    s = single(s);                  % [1, K]

    A_smooth = A ./ s;               % [M, K]
    B_smooth = B .* s.';            % [K, N]   (s.' is [K, 1])

    C_ref = A * B;
    C_smooth_ref = A_smooth * B_smooth;
    terr = error_metrics(C_smooth_ref, C_ref, eps);

    info = struct();
    info.alpha = double(alpha);
    info.scale_min = double(min(s));
    info.scale_max = double(max(s));
    info.scale_mean = double(mean(s));
    info.scale_std = double(std(s));
    info.transform_fro_relative_error = terr.fro_relative_error;
    info.transform_max_abs_error = terr.max_abs_error;
    info.clamp_scale = clamp_scale;
end