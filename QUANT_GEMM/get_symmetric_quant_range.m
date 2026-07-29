function [qmin, qmax] = get_symmetric_quant_range(num_bits)
    % Symmetric signed quantization range.
    %   4-bit:  [-7, 7],  8-bit: [-127, 127],  16-bit: [-32767, 32767]
    % We intentionally avoid -128 for int8 symmetric quantization.
    if num_bits <= 1
        error('get_symmetric_quant_range:num_bits', ...
              'num_bits must be greater than 1 for symmetric signed quantization.');
    end
    qmax = 2 ^ (num_bits - 1) - 1;
    qmin = -qmax;
end