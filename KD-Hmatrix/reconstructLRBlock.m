function A = reconstructLRBlock(H)

    m = numel(H.row);
    n = numel(H.col);

    A = zeros(m,n);

    for k = 1:numel(H.groups)

        Uk = H.groups(k).U;
        Vk = H.groups(k).V;

        b = H.groups(k).bits;

        A = A + gemm_AB(Uk, Vk', b);

    end

end