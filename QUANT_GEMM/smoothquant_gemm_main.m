%% smoothquant_gemm.m
%  MATLAB translation of the Python quantization / SmoothQuant GEMM emulation library.
%
%  Symmetric integer quantization (per-tensor and per-channel), integer GEMM
%  accumulation emulation, SmoothQuant activation/weight smoothing, and
%  baseline-vs-SmoothQuant comparison sweeps.
%
%  Run as a script: the main demo executes, then local functions are available.

%% ----------------------------- main demo -----------------------------
M = 1024; K = 100; N = 1024;

% Case 1: ordinary random normal.
results = sweep_bitwidths_with_smoothquant( ...
    M, K, N, ...
    'bitwidths', [4 6 8 10 12 16], ...
    'alpha', 0.5, ...
    'seed', 0, ...
    'inject_activation_outliers', false);
print_bitwidth_smoothquant_table(results);

% Case 2: outlier-heavy activation (SmoothQuant usually helps here).
results_outlier = sweep_bitwidths_with_smoothquant( ...
    M, K, N, ...
    'bitwidths', [4 6 8 10 12 16], ...
    'alpha', 0.5, ...
    'seed', 0, ...
    'inject_activation_outliers', true, ...
    'outlier_scale', 20.0, ...
    'outlier_fraction', 0.01);

fprintf('\nWith activation outliers\n');
print_bitwidth_smoothquant_table(results_outlier);

% Alpha sweep on the outlier case.
alpha_results = sweep_smoothquant_alpha( ...
    results_outlier.A, results_outlier.B, ...
    'num_bits', 8, ...
    'alphas', [0.0 0.25 0.5 0.75 1.0]);

fprintf('\n');
print_alpha_sweep_table(alpha_results);

sanity_check_smoothquant_identity();
sanity_check_integer_path();





function out = compare_baseline_and_smoothquant(A, B, num_bits, alpha, eps, clamp_scale)
    % Compare baseline symmetric quantized GEMM vs SmoothQuant.
    if nargin < 3, num_bits = 8; end
    if nargin < 4, alpha = 0.5; end
    if nargin < 5, eps = 1e-12; end
    if nargin < 6, clamp_scale = []; end

    baseline_s1 = gemm_symmetric_A_pertensor_B_perchannel(A, B, num_bits, eps);
    baseline_s2 = gemm_symmetric_A_perchannel_B_perchannel(A, B, num_bits, eps);
    smooth_s1   = gemm_smoothquant_A_pertensor_B_perchannel(A, B, num_bits, alpha, eps, clamp_scale);
    smooth_s2   = gemm_smoothquant_A_perchannel_B_perchannel(A, B, num_bits, alpha, eps, clamp_scale);

    out = struct();
    out.baseline_strategy_1_A_pertensor_B_perchannel = baseline_s1;
    out.baseline_strategy_2_A_perchannel_B_perchannel = baseline_s2;
    out.smoothquant_strategy_1_A_pertensor_B_perchannel = smooth_s1;
    out.smoothquant_strategy_2_A_perchannel_B_perchannel = smooth_s2;

    out.summary = struct();
    out.summary.num_bits = num_bits;
    out.summary.alpha = alpha;
    out.summary.baseline_s1_rel = baseline_s1.relative_error;
    out.summary.baseline_s2_rel = baseline_s2.relative_error;
    out.summary.smooth_s1_rel = smooth_s1.relative_error;
    out.summary.smooth_s2_rel = smooth_s2.relative_error;
    out.summary.baseline_s1_rmse = baseline_s1.error_metrics.rmse;
    out.summary.baseline_s2_rmse = baseline_s2.error_metrics.rmse;
    out.summary.smooth_s1_rmse = smooth_s1.error_metrics.rmse;
    out.summary.smooth_s2_rmse = smooth_s2.error_metrics.rmse;
    out.summary.smooth_s1_improvement_ratio = ...
        baseline_s1.relative_error / (smooth_s1.relative_error + eps);
    out.summary.smooth_s2_improvement_ratio = ...
        baseline_s2.relative_error / (smooth_s2.relative_error + eps);
end

function out = sweep_bitwidths_with_smoothquant(M, K, N, varargin)
    % Generate random A, B and compare baseline vs SmoothQuant across bitwidths.
    % Name-value options: bitwidths, alpha, seed, eps, clamp_scale,
    %   inject_activation_outliers, outlier_scale, outlier_fraction.
    p = inputParser;
    addParameter(p, 'bitwidths', [4 6 8 10 12 16]);
    addParameter(p, 'alpha', 0.5);
    addParameter(p, 'seed', 0);
    addParameter(p, 'eps', 1e-12);
    addParameter(p, 'clamp_scale', []);
    addParameter(p, 'inject_activation_outliers', false);
    addParameter(p, 'outlier_scale', 20.0);
    addParameter(p, 'outlier_fraction', 0.01);
    parse(p, varargin{:});
    opt = p.Results;

    rng(opt.seed);
    % Generate in half precision then promote to single (mimics the Python path).
    A = cast_maybe_half(randn(M, K));
    B = cast_maybe_half(randn(K, N));
    A = single(A);
    B = single(B);

    if opt.inject_activation_outliers
        numelA = numel(A);
        num_outliers = max(1, floor(numelA * opt.outlier_fraction));
        idx = randperm(numelA, num_outliers);
        A(idx) = A(idx) * opt.outlier_scale;
    end

    % NOTE: MATLAB field names cannot start with a digit, so we store the
    % per-bitwidth results in a struct array (keyed by a numeric .bits field)
    % instead of using dynamic field names like '4', '6', etc.
    fields = {'bits','baseline_s1_rel','baseline_s2_rel','smooth_s1_rel', ...
              'smooth_s2_rel','baseline_s1_rmse','baseline_s2_rmse', ...
              'smooth_s1_rmse','smooth_s2_rmse', ...
              'smooth_s1_improvement_ratio','smooth_s2_improvement_ratio'};
    results_by_bits = struct('bits', {}, 'result', {});
    emptyS = struct();
    for i = 1:numel(fields), emptyS.(fields{i}) = []; end
    summary_by_bits = repmat(emptyS, 0);

    for i = 1:numel(opt.bitwidths)
        bits = opt.bitwidths(i);
        result = compare_baseline_and_smoothquant(A, B, bits, opt.alpha, opt.eps, opt.clamp_scale);
        results_by_bits(i).bits = bits;
        results_by_bits(i).result = result;
        s = result.summary;
        summary_by_bits(i).bits = bits;
        summary_by_bits(i).baseline_s1_rel = s.baseline_s1_rel;
        summary_by_bits(i).baseline_s2_rel = s.baseline_s2_rel;
        summary_by_bits(i).smooth_s1_rel   = s.smooth_s1_rel;
        summary_by_bits(i).smooth_s2_rel   = s.smooth_s2_rel;
        summary_by_bits(i).baseline_s1_rmse = s.baseline_s1_rmse;
        summary_by_bits(i).baseline_s2_rmse = s.baseline_s2_rmse;
        summary_by_bits(i).smooth_s1_rmse   = s.smooth_s1_rmse;
        summary_by_bits(i).smooth_s2_rmse   = s.smooth_s2_rmse;
        summary_by_bits(i).smooth_s1_improvement_ratio = s.smooth_s1_improvement_ratio;
        summary_by_bits(i).smooth_s2_improvement_ratio = s.smooth_s2_improvement_ratio;
    end

    out = struct();
    out.A = A;
    out.B = B;
    out.results_by_bits = results_by_bits;
    out.summary_by_bits = summary_by_bits;
    out.alpha = opt.alpha;
    out.inject_activation_outliers = opt.inject_activation_outliers;
    out.outlier_scale = opt.outlier_scale;
    out.outlier_fraction = opt.outlier_fraction;
end

function x = cast_maybe_half(x)
    % Cast through half precision if the 'half' type is available; else no-op.
    try
        x = cast(x, 'half');
    catch
        % half type unavailable; leave as-is (double from randn)
    end
end

function out = sweep_smoothquant_alpha(A, B, varargin)
    % Sweep SmoothQuant alpha for a fixed bitwidth.
    p = inputParser;
    addParameter(p, 'num_bits', 8);
    addParameter(p, 'alphas', [0.0 0.25 0.5 0.75 1.0]);
    addParameter(p, 'eps', 1e-12);
    addParameter(p, 'clamp_scale', []);
    parse(p, varargin{:});
    opt = p.Results;

    % Use a struct array keyed by a numeric .alpha field (field names cannot
    % start with a digit, so '0', '0.25' are illegal as dynamic field names).
    rows = struct('alpha', {}, 'summary', {});
    for i = 1:numel(opt.alphas)
        alpha = opt.alphas(i);
        result = compare_baseline_and_smoothquant(A, B, opt.num_bits, alpha, opt.eps, opt.clamp_scale);
        rows(i).alpha = alpha;
        rows(i).summary = result.summary;
    end

    out = struct();
    out.num_bits = opt.num_bits;
    out.alphas = opt.alphas;
    out.rows = rows;
end

function print_bitwidth_smoothquant_table(results)
    fprintf('Relative forward error: baseline vs SmoothQuant\n');
    fprintf('%s\n', repmat('=', 1, 150));
    fprintf('%6s | %12s | %12s | %12s | %12s | %12s | %12s\n', ...
        'bits', 'base s1', 'smooth s1', 'improve s1', 'base s2', 'smooth s2', 'improve s2');
    fprintf('%s\n', repmat('-', 1, 150));

    sb = results.summary_by_bits;
    for i = 1:numel(sb)
        b = sb(i).bits;
        row = sb(i);
        fprintf('%6d | %12.6e | %12.6e | %12.6f | %12.6e | %12.6e | %12.6f\n', ...
            b, row.baseline_s1_rel, row.smooth_s1_rel, row.smooth_s1_improvement_ratio, ...
            row.baseline_s2_rel, row.smooth_s2_rel, row.smooth_s2_improvement_ratio);
    end

    fprintf('%s\n', repmat('-', 1, 150));
    fprintf('SmoothQuant alpha: %g\n', results.alpha);
    fprintf('Activation outliers injected: %d\n', results.inject_activation_outliers);
end

function print_alpha_sweep_table(results)
    fprintf('SmoothQuant alpha sweep\n');
    fprintf('%s\n', repmat('=', 1, 120));
    fprintf('%8s | %12s | %12s | %12s | %12s | %12s | %12s\n', ...
        'alpha', 'base s1', 'smooth s1', 'improve s1', 'base s2', 'smooth s2', 'improve s2');
    fprintf('%s\n', repmat('-', 1, 120));

    for i = 1:numel(results.rows)
        row = results.rows(i);
        alpha = row.alpha;
        s = row.summary;
        fprintf('%8.3f | %12.6e | %12.6e | %12.6f | %12.6e | %12.6e | %12.6f\n', ...
            alpha, s.baseline_s1_rel, s.smooth_s1_rel, s.smooth_s1_improvement_ratio, ...
            s.baseline_s2_rel, s.smooth_s2_rel, s.smooth_s2_improvement_ratio);
    end
    fprintf('%s\n', repmat('-', 1, 120));
end

function sanity_check_smoothquant_identity()
    % Verify  A*B == (A/s)*(B*s)  up to FP rounding.
    rng(123);
    A = single(randn(1024, 100));
    B = single(randn(100, 1024));
    [A_s, B_s, s, ~] = smoothquant_transform(A, B, 0.5);

    C0 = A * B;
    C1 = A_s * B_s;
    metrics = error_metrics(C1, C0);

    fprintf('\nSanity check: SmoothQuant identity\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('scale min: %g\n', min(s));
    fprintf('scale max: %g\n', max(s));
    fprintf('transform relative error: %g\n', metrics.fro_relative_error);
    fprintf('transform max abs error: %g\n', metrics.max_abs_error);
end

function sanity_check_integer_path()
    % Check integer path equals dequantized matmul up to FP rounding:
    %   Acc = Q_A @ Q_B ;  C = Acc * scale_A * scale_B
    % versus  dequant(Q_A) @ dequant(Q_B).
    rng(456);
    A = single(randn(1024, 100));
    B = single(randn(100, 1024));

    out = gemm_symmetric_A_pertensor_B_perchannel(A, B, 16);
    Q_A = out.A_quant;
    Q_B = out.B_quant;

    A_deq = dequantize_symmetric(Q_A, out.params.A);
    B_deq = dequantize_symmetric(Q_B, out.params.B);

    C_deq_matmul = A_deq * B_deq;
    C_integer_path = out.C;

    err_1 = norm(C_deq_matmul-A*B,"fro")/norm(A*B,"fro");
    err_2 = norm(C_integer_path-A*B,"fro")/norm(A*B,"fro");

    fprintf("Comparison with fp32\n");
    disp([err_1, err_2]);
    
 
    max_diff = max(abs(C_deq_matmul(:) - C_integer_path(:)));

    fprintf('\nSanity check: integer GEMM path\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('A_quant dtype: %s\n', class(Q_A));
    fprintf('B_quant dtype: %s\n', class(Q_B));
    fprintf('Accumulator dtype: %s\n', class(out.accumulator));
    fprintf('Max diff vs dequant(A) @ dequant(B): %g\n', max_diff);
end
