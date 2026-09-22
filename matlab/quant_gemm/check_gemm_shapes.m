function check_gemm_shapes(A, B)
    if ndims(A) ~= 2 || ndims(B) ~= 2
        error('check_gemm_shapes:dim', 'A and B must both be 2D matrices.');
    end
    if size(A, 2) ~= size(B, 1)
        error('check_gemm_shapes:mismatch', ...
              'Shape mismatch: A %s, B %s. Expected size(A,2)==size(B,1).', ...
              mat2str(size(A)), mat2str(size(B)));
    end
end