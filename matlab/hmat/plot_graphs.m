clear; clc; close all;

set(gcf,'PaperPositionMode','auto');
set(gcf,'Color','w');
%% Define consistent colors
c_fp64 = [1 0 0];          % red
c_fp32 = [1 0.6 0];        % orange
c_fp16 = [0.49 0.18 0.56]; % purple

%% Data
x1 = [1e-12 1e-10 1e-8 1e-6 1e-4 1e-2 1e-1]; % Tolerance (epsilon)

% Global approximation error (N=8000, L=2)
e_f64 = [4.0352e-15 4.2264e-13 3.9629e-10 7.3375e-09 8.4573e-06 5.4694e-04 0.0070];  % amp Hybrid 
e_f32 = [2.7009e-08 2.7031e-08 2.7231e-08 2.7945e-08 8.4574e-06 5.4694e-04 0.0070];  % amp Hybrid 
e_f16 = [2.1250e-04 2.1028e-04 2.1381e-04 2.1324e-04 2.1271e-04 5.8636e-04 0.0070];  % amp Hybrid


% % Global approximation error (N=64000, L=3)
% e_f64 = [4.9463e-15 1.5274e-13 1.4388e-09 3.0630e-09 1.8435e-05 0.0014 0.0030];  % amp Hybrid 
% e_f32 = [6.5636e-08 6.5533e-08 6.5736e-08 6.5695e-08 1.8436e-05 0.0014 0.0030];  % amp Hybrid 
% e_f16 = [5.2600e-04 5.2603e-04 5.2655e-04 5.2563e-04 5.2697e-04 0.0015 0.0031];  % amp Hybrid
% 
dim = 3;
level = 2;  % L = 2, 3
hodlr_level = level-1;
eta = sqrt(dim);
c1 = (2^dim-1)*(1+(2*sqrt(dim)/eta))^dim;
c2 = (1+(2*sqrt(dim)/eta))^dim - 1;
c3 = (2^dim-1);
Bound_hybrid = 2*(2*sqrt(hodlr_level*c1 + c2 + (level-hodlr_level)*c3) + 1);
Bound_hmat = 2*sqrt(level*c1 + c2) + 1;


% Error in MVP

loglog(x1, e_f64, '-d', ...
    'Color', c_fp64, ...
    'LineWidth', 2, ...
    'MarkerSize', 10, ...
    'DisplayName', '$\mathrm{fp64}$');
hold on;

loglog(x1, e_f32, '-o', ...
    'Color', c_fp32, ...
    'LineWidth', 2, ...
    'MarkerSize', 10, ...
    'DisplayName', '$\mathrm{fp32}$');

loglog(x1, e_f16, '-s', ...
    'Color', c_fp16, ...
    'LineWidth', 2, ...
    'MarkerSize', 10, ...
    'DisplayName', '$\mathrm{fp16}$');

% Plot Bound
loglog(x1, Bound_hybrid*x1, 'k--', 'LineWidth', 2, ...
    'DisplayName', '$\mathrm{eb}$');


%%%%%%%%%%%%%%%%%%%%%

%% Labels & Grid
% xlabel('$\epsilon$ (target accuracy)','Interpreter','latex','FontSize', 50, 'FontWeight','bold');
% ylabel('Global construction error','Interpreter','latex','FontSize',50);

grid on;
ax = gca;

ax.XScale = 'log';
ax.YScale = 'log';

ax.XLim = [1e-12 1e-1];
ax.YLim = [1e-16 1];

ax.XTick = 10.^(-12:2:-1);
ax.YTick = 10.^(-15:2:0);

ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.1;
ax.FontSize = 17;
ax.LineWidth = 1;

ax.XLabel.FontSize = 25;   % increase only xlabel
ax.YLabel.FontSize = 25;   % increase only ylabel

% optional: bold
ax.XLabel.FontWeight = 'bold';
ax.YLabel.FontWeight = 'bold';

ax.TickLabelInterpreter = 'latex';

%% Legend
lgd = legend('Interpreter','latex','Location','southeast');
set(lgd,'Interpreter','latex','FontSize',20,'Box','on', 'Color','white');

%% Export
set(gcf,'Renderer','painters');
exportgraphics(gcf,'figures/plot_mvp_error.pdf', ...
    'ContentType','vector');

