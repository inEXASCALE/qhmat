clear;
clc;
close all;

addpath('/Users/riteshkhan/Documents/quantH/QUANT_GEMM');


%%%%%%%%%%%%%%%%%%%%%% Parameters of H-matrix %%%%%%%%%%%%%%%%%%%%%%%

N = 1000;          % Number of particles, generates N x N kernel matrix
d = 2;             % dimension of the underlying space

minBlockSize = 64;

eta = 2*sqrt(d);   % Admissibility parameter

LR_epsilon = 1e-3; % Low-rank tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


rng(100);
%% Generate Different points 

%% Random particle.
particle_loc = rand(N,d);


%%% Points are on the boundary of circle.
% theta = linspace(0, 2*pi, N+1).';
% theta(end) = [];
% particle_loc = [cos(theta), sin(theta)];


%%% Points are on the boundary of star shape.
% theta = linspace(0,2*pi,N+1).';
% theta(end) = [];
% r = 1 + 0.25*cos(5*theta) + 0.10*cos(10*theta);
% particle_loc = [r.*cos(theta), r.*sin(theta)];


%%% Points are on the surface of a sphere.
% i = (0:N-1).';
% golden_angle = pi * (3 - sqrt(5));
% z = 1 - 2*(i + 0.5)/N;
% r = sqrt(max(0, 1 - z.^2));
% theta = golden_angle * i;
% particle_loc = [r .* cos(theta), r .* sin(theta), z];



%%% Pick a kernel function

kernel_choice = 2;

switch kernel_choice
    case 1
        % Logarithmic kernel
        kernelFun = @(x,y) logKernel(x,y);

    case 2
        % 1/r kernel
        kernelFun = @(x,y) inverseDistanceKernel(x,y);

    case 3
        % Matern kernel
        kernelFun = @(x,y) expKernel(x,y);

    otherwise
        error('Unknown kernel_choice.');
end

%%%%%%%%%% Different Kernel functions  %%%%%%%%%%
function k = logKernel(x,y)
    r = norm(x-y);

    if r <= eps
        k = 1;
    else
        k = log(r);
    end
end

function k = inverseDistanceKernel(x,y)
    r = norm(x-y);

    if r <= eps
        k = 1;
    else
        k = 1/r;
    end
end

function k = expKernel(x,y)
    r = norm(x-y);
    k = exp(-r);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Bit width set
bit_set = [2 4 6 8 10 12 14 16];


%% Create H-matrix object
H1 = HmatKD(particle_loc, minBlockSize, kernelFun, eta, LR_epsilon, bit_set);



%% Build cluster tree
H1.buildClusterTree();


%% Get max-norm of the matrix
H1.getMaxnorm();

%% Build H-matrix
[~, stats] = H1.buildHMatrix();

totalStorageFWPH = stats.lowRankStorageFWP + stats.denseStorage;
totalStorageQuantH = stats.lowRankStorageQuant + stats.denseStorage;

fullDense = N * N * 64;
gain1 = fullDense / totalStorageFWPH;
gain2 = totalStorageFWPH/totalStorageQuantH;

fprintf('Parameters\n\n');
fprintf('Number of particles                  : %d\n', N);
fprintf('Minimum block size                   : %d\n', minBlockSize);
fprintf('Underlying dimension                 : %d\n', d);
fprintf('Admissibility parameter (eta)        : %f\n', eta);

fprintf('\nStorage information\n\n');
fprintf('Low-rank tolerance (epsilon)         : %0.e\n', LR_epsilon);
fprintf('Low-rank storage (Working precision) : %d\n', stats.lowRankStorageFWP);
fprintf('Low-rank storage (Quantized version) : %d\n', stats.lowRankStorageQuant);
fprintf('Dense storage (Digonal blocks)       : %d\n', stats.denseStorage);
fprintf('Total storage (Working precision)    : %d\n', totalStorageFWPH);
fprintf('Total storage (Quantized version)    : %d\n', totalStorageQuantH);
% fprintf('Full dense matrix size (N x N)       : %d\n', fullDense);
% fprintf('Storage gain (Dense vs H_wp)  : %.2f x\n', gain1);
fprintf('Storage gain in quantized format (H_wp vs H_quat)   = %.2f x\n', gain2);

fprintf('\nAccuracy information\n\n');

[relErrQ, ~, ~] = H1.reconstructionErrorQuantH(H1.Hroot);

fprintf('Relative error in re-constructed quantized H-matrix = %.3e\n', relErrQ);


%% Random dense vector/matrix multiplied with H-matrix format
x = randn(N,25);

%% H-matrix-vector multiplication
y1=H1.HmatVec(H1.Hroot,x);

% MVP Accuracy test
[relErr,~,yH,yFull] = H1.testHmatVec(H1.Hroot,x);


fprintf('Relative error in quantized H-matrix-vector product = %.3e\n',relErr);




%% Visualization (domain, H-matrix, etc) (optional)
H1.visualizeOriginal();
H1.visualizePermutationPattern();
% H1.visualizeAdmissibility();
H1.visualizeHMatrixBlocks();

if(d<=2)
    H1.plotKDTree();
end