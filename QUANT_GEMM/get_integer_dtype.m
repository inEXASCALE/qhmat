function qdtype = get_integer_dtype(num_bits)
    % Choose signed integer dtype string for storing symmetric quantized values.
    if num_bits <= 8
        qdtype = 'int8';
    elseif num_bits <= 16
        qdtype = 'int16';
    elseif num_bits <= 32
        qdtype = 'int32';
    else
        qdtype = 'int64';
    end
end