function r = npround(x)
    % Round half to even (matches numpy.round). Vectorized.
    r = floor(x);
    f = x - r;
    up = f > 0.5;
    half = f == 0.5;
    r(up) = r(up) + 1;
    tieup = half & (mod(r, 2) ~= 0);   % tie: go to even
    r(tieup) = r(tieup) + 1;
end