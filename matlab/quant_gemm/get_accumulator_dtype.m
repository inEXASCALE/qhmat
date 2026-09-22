function adtype = get_accumulator_dtype(qdtype)
    % Accumulator dtype (string) for quantized GEMM emulation.
    %   int8  -> int32
    %   int16 -> int64
    %   int32 -> int64
    %   int64 -> int64
    % No per-step saturating accumulation is used.
    switch qdtype
        case 'int8'
            adtype = 'int32';
        case 'int16'
            adtype = 'int64';
        case 'int32'
            adtype = 'int64';
        case 'int64'
            adtype = 'int64';
        otherwise
            error('get_accumulator_dtype:unsupported', 'Unsupported quantized dtype: %s', qdtype);
    end
end