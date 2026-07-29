function w = get_bit_width(U, V, tol, hnorm)
    umax = max(abs(U), [], 'all');
    vmax = max(abs(V), [], 'all');
    uinf = norm(U, 'inf');
    v1 = norm(V, 1);
    [~, k] = size(U);

    m = (uinf+v1);
    alpha = max(umax,vmax);

    A = tol * hnorm;
    B = -((4 * tol * hnorm) + alpha * m); 
    C = 4 * tol * hnorm + (2 * alpha * m) - k * (alpha * alpha);
    w = log2((-B + sqrt(B*B- 4*A*C))/(2*A));
    w = 2*ceil(w/2);
end