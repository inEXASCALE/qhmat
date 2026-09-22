function plotBitwidthMap(precisionMap, bitWidths)

    %--------------------------------------------------------------
    % Visualize bit-width assignment across singular-value indices
    %
    % precisionMap : 1 x r vector containing assigned bit widths
    % bitWidths    : available bit widths, e.g.
    %                [2 4 6 8 10 12 14 16 32]
    %--------------------------------------------------------------

    bitWidths = sort(bitWidths(:).');

    % Map actual bit widths -> integer color indices
    map = zeros(size(precisionMap));

    for k = 1:numel(bitWidths)
        map(precisionMap == bitWidths(k)) = k;
    end

    % Check for invalid/unassigned entries
    if any(map(:) == 0)
        warning('Some entries of precisionMap do not match bitWidths.');
    end

    %--------------------------------------------------------------
    % Plot
    %--------------------------------------------------------------

    figure;

    imagesc(map(:).');

    % One color for each available bit width
    colormap(turbo(numel(bitWidths)));

    clim([0.5, numel(bitWidths)+0.5]);

    %--------------------------------------------------------------
    % Colorbar
    %--------------------------------------------------------------

    cb = colorbar;

    cb.Ticks = 1:numel(bitWidths);
    cb.TickLabels = arrayfun(@(b) sprintf('%d-bit',b), ...
                             bitWidths, ...
                             'UniformOutput',false);

    %--------------------------------------------------------------
    % Axes
    %--------------------------------------------------------------

    yticks([]);

    xlabel('Singular value index');
    title('Adaptive bit-width assignment');

end