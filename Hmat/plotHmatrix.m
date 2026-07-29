function plotHmatrix(H)
fig = figure;
hold on;
axis equal;
axis off;
% xlabel('Columns'); ylabel('Rows');

N = H.N;

% Loop over H matrix levels
for j = 2:(H.nLevels - H.hodlr_levels + 1)
    nBoxes = H.nBoxesPerLevel(j);
    diam = N/nBoxes;
    for k = 1:nBoxes
        boxK = H.tree{j}{k};
        I_list = boxK.interactionList;
        for idx = 1:numel(I_list)
            ki = I_list(idx);
            x = (k-1)*diam;
            y = N-(ki)*diam;
            rectangle('Position', [x, y, diam, diam], ...
                'FaceColor', [0 1 0], ...
                'EdgeColor', 'black', ...
                'LineWidth', 0.5);
            [~, r] = size(boxK.U{ki});
            % Show rank
            % text(x + diam/2, y + diam/2, num2str(r), ...
            %     'HorizontalAlignment', 'center', ...
            %     'VerticalAlignment', 'middle', ...
            %     'FontSize', 8, ...
            %     'Color', 'k');
        end
    end
end

% Junction level (NBD)
j = H.nLevels-H.hodlr_levels+1;
nBoxes = H.nBoxesPerLevel(j);
diam = N/nBoxes;
for k = 1:nBoxes
    boxK = H.tree{j}{k};
    % --- NEIGHBORS (orange) ---
    for idx = 1:numel(boxK.neighborNumbers)
        nn = boxK.neighborNumbers(idx);
        x = (k-1)*diam;
        y = N-(nn)*diam;
        if(H.hodlr_levels>0)
            rectangle('Position', [x, y, diam, diam], ...
                'FaceColor', [0.0 0.45 0.74], ...
                'EdgeColor', 'black', ...
                'LineWidth', 0.5);                % Orange: [1 0.5 0]
            [~, r] = size(boxK.U{nn});
            % Show rank
            % text(x + diam/2, y + diam/2, num2str(r), ...
            %     'HorizontalAlignment', 'center', ...
            %     'VerticalAlignment', 'middle', ...
            %     'FontSize', 8, ...
            %     'Color', 'k');
        else
            rectangle('Position', [x, y, diam, diam], ...
                'FaceColor', [1 0.5 0], ...
                'EdgeColor', 'black', ...
                'LineWidth', 0.5);                % Orange: [1 0.5 0]
        end
    end
end

% Loop over HODLR matrix levels
for j = (H.nLevels-H.hodlr_levels+2):(H.nLevels+1)
    nBoxes = H.nBoxesPerLevel(j);
    diam = N/nBoxes;
    for k = 1:nBoxes
        boxK = H.tree{j}{k};
        pN = ceil(k / H.pow2d);
        for c = 1:H.pow2d
            ki = H.tree{j-1}{pN}.childrenNumbers(c);
            % disp([k ki]);
            if(ki ~= k)
                x = (k-1)*diam;
                y = N-(ki)*diam;
                rectangle('Position', [x, y, diam, diam], ...
                    'FaceColor', [0.0 0.45 0.74], ...
                    'EdgeColor', 'black', ...
                    'LineWidth', 0.5);
                [~, r] = size(boxK.U{ki});
                % Show rank
                % text(x + diam/2, y + diam/2, num2str(r), ...
                %     'HorizontalAlignment', 'center', ...
                %     'VerticalAlignment', 'middle', ...
                %     'FontSize', 8, ...
                %     'Color', 'k');
            end
        end
    end
end

% Leaf level (Self)
x = 0;
y = N-diam;
for k = 1:nBoxes
    % % Show rank
    % rank_val = length(I);  % self block full rank
    % drawRankText(i_min, i_max, i_min, i_max, rank_val);
    % --- Self (Red) ---
    % disp([x y diam]);
    rectangle('Position', [x, y, diam, diam], ...
        'FaceColor', [1 0 0], ...
        'EdgeColor', 'black', ...
        'LineWidth', 0.5);

    x = x + diam;
    y = y - diam;
end


xlim([1 N]); ylim([1 N]);
% Apply tight layout
set(gca, 'LooseInset', get(gca, 'TightInset'));  % Minimize margins around axes
exportgraphics(gcf,'figures/hmat_fig.pdf', ...
    'ContentType','vector');
% hold off;
% exportgraphics(gca, 'figures/hmat_fig.pdf')

% close(fig);

% title('Hierarchical Matrix Block Structure with Ranks');
end

% function drawRankText(i_min,i_max,j_min,j_max,rank_val)
%     % Draw rank in the center of the rectangle
%     x = (j_min + j_max)/2;
%     y = (i_min + i_max)/2;
%     text(x, y, num2str(rank_val), 'HorizontalAlignment','center', ...
%         'VerticalAlignment','middle', 'FontSize',8, 'Color','k');
% end
