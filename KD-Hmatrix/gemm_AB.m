function C = gemm_AB(A, B, bitWidth)
    out = gemm_smoothquant_A_perchannel_B_perchannel(A, B, bitWidth);
    C = out.C;
end