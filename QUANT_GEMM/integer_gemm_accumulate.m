function Acc = integer_gemm_accumulate(A_q, B_q, accumulator_dtype)
    % Integer GEMM accumulation:  Acc(m,n) = sum_k A_q(m,k)*B_q(k,n).
    % No dequantization inside the dot product.
    %
    % NOTE: MATLAB does not provide integer matrix multiplication (mtimes on
    % integer types is unsupported). Because the Python reference uses
    % non-saturating accumulation in a wider integer type, and the test
    % sizes never overflow that accumulator, the exact same result is obtained
    % by promoting to double, multiplying, and casting back to the accumulator
    % integer type. This is faithful whenever no accumulator overflow occurs.
    if ndims(A_q) ~= 2 || ndims(B_q) ~= 2
        error('integer_gemm_accumulate:dim', 'A_q and B_q must both be 2D matrices.');
    end
    if size(A_q, 2) ~= size(B_q, 1)
        error('integer_gemm_accumulate:mismatch', ...
              'Shape mismatch: A_q %s, B_q %s.', mat2str(size(A_q)), mat2str(size(B_q)));
    end
    if ~isinteger(A_q)
        error('integer_gemm_accumulate:typeA', 'A_q must be integer.');
    end
    if ~isinteger(B_q)
        error('integer_gemm_accumulate:typeB', 'B_q must be integer.');
    end
    if ~ismember(accumulator_dtype, {'int8','int16','int32','int64'})
        error('integer_gemm_accumulate:acc', 'accumulator_dtype must be integer.');
    end

    A_acc = cast(A_q, accumulator_dtype);
    B_acc = cast(B_q, accumulator_dtype);

    Acc = cast(double(A_acc) * double(B_acc), accumulator_dtype);
end