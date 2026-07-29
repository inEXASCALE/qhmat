function m = error_metrics(C_hat, C_ref, eps)
    if nargin < 3, eps = 1e-12; end
    C_hat = single(C_hat);
    C_ref = single(C_ref);
    diff = C_hat - C_ref;

    fro_abs = norm(diff, 'fro');
    fro_ref = norm(C_ref, 'fro');

    m = struct();
    m.fro_absolute_error = double(fro_abs);
    m.fro_relative_error = double(fro_abs / (fro_ref + eps));
    m.rmse = double(sqrt(mean(diff(:) .* diff(:))));
    m.mae = double(mean(abs(diff(:))));
    m.max_abs_error = double(max(abs(diff(:))));
    m.rms_ref = double(sqrt(mean(C_ref(:) .* C_ref(:))));
    m.mean_abs_ref = double(mean(abs(C_ref(:))));
    m.max_abs_ref = double(max(abs(C_ref(:))));
end