function y = quant_block_mvp(groups,x)

    m = size(groups(1).U,1);

    y = zeros(m,size(x,2));

    for k = 1:numel(groups)

        Uk = groups(k).U;
        Vk = groups(k).V;

        b = groups(k).bits;

        % V_k' * x
        tmp = gemm_AB(Vk', x, b);

        % U_k * tmp
        y = y + gemm_AB(Uk, tmp, b);

    end

end