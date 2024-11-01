function [A, B, C, const, output] = btd_ll1_als_3d(T, R, Lr, options, varargin)
% BTD_LL1_ALS_3D Computes the (Lr,Lr,1) block-term decomposition (BTD) of a
% 3rd-order tensor using the alternative least squares algorithm (ALS).
%
% INPUT:
%   T (I_1 x I_2 x I_3): 3-D tensor data tensor
%   R (1 x 1): number of BTD components (sources)
%   Lr (1 x R): Ranks of the factor matrices in the first and second mode for
%               all components
%   options (struct): optimization options including:
%          - th_relerr (1 x 1): relative error threshold
%          - maxiter   (1 x 1): max number of iterations
%   variable extra inputs (1 x 3 cell array OR 3 arrays, OPTIONAL): 
%           initialization for the factor matrices
%
% OUTPUT:
%   A (I_1 x sum(Lr)): mode-1 factor matrix
%   B (I_2 x sum(Lr)): mode-2 factor matrix 
%   C (I_3 x R): mode-3 factor matrix with normalized columns 
%   const (1 x R):  vector containing the respective weights for the BTD
%           components
%   output (struct) : optimization options including:
%          - numiter (1 x 1): the number of iterations the algorithm ran for
%          - relerr (1 x numiter): relative error achieved, defined as 
%           Frobenius norm of the residual of the decomposition OVER 
%           Frobenius norm of the original tensor.
% 
% 

% Get algorithm parameters
% Number of iterations
maxiter = options.maxiter;
% Threshold for relative error
th_relerr = options.th_relerr;

% Check if the initialization for the factor matrices was given
init = [];
if ~isempty(varargin)
    if length(varargin) == 1    % Given as cell
        init = varargin{:}; 
    else                        % Given as matrices 
        init = varargin;
    end
end

% Initialize the three factor matrices 
if isempty(init)    % Randomly if not initialization was given
    A = randn(size(T, 1), sum(Lr));
    B = randn(size(T, 2), sum(Lr));
    C = randn(size(T, 3), R);
else                % Otherwise use the given initialization
    A = init{1};
    B = init{2};
    C = init{3};
end

% Normalize the columns of the initial factor matrices
A = A ./ vecnorm(A);
B = B ./ vecnorm(B);
C = C ./ vecnorm(C);

% Obtain the three tensor unfoldings 
T1 = hidden_mode_n_matricization(T,1);
T2 = hidden_mode_n_matricization(T,2);
T3 = hidden_mode_n_matricization(T,3);

% ALS iterations
% A_sizes = size(A);
% B_sizes = size(B);
% C_sizes = size(C);
% prods_mode1 = zeros(C_sizes(1) * B_sizes(1), R * B_sizes(2));
% prods_mode2 = zeros(C_sizes(1) * A_sizes(1), R * A_sizes(2));
khatri_rao_prods = zeros(size(A,1) * size(B,1), R); % I_1*I_2 x R
for idxiter = 1:maxiter
    disp(idxiter)
    % Mode 1 (do not forget to normalize the columns!)
    prods = [];
    idx_start = 1;
    for i = 1:R
        idx_end = sum(Lr(1:i));
        prods = [prods kron(C(:, i), B(:, idx_start:idx_end))];
        idx_start = idx_end+1;
    end
    A = T1 * pinv(prods');
%     A = A ./ vecnorm(A);
    % Mode 2 (do not forget to normalize the columns!)
    prods = [];
    idx_start = 1;
    for i = 1:R
        idx_end = sum(Lr(1:i));
        prods = [prods kron(C(:, i), A(:, idx_start:idx_end))];
        idx_start = idx_end+1;
    end
    B = T2 * pinv(prods');
%     B = B ./ vecnorm(B);
    % Mode 3 (do not forget to normalize the columns!)
    idx_start = 1;
    for i = 1:R
        idx_end = sum(Lr(1:i));
        khatri_rao_prods(:, i) = sum(hidden_khatri_rao(B(:, idx_start:idx_end), A(:, idx_start:idx_end)), 2);
        idx_start = idx_end+1;
    end
    C = T3 * pinv(khatri_rao_prods');
    const = vecnorm(C);
    C = C ./ const;
    % Compute the current estimate of the mode-3 unfolding of the tensor
    T3_est = C * diag(const) * khatri_rao_prods';

    % Calculate the relative error between the estimate and the true
    % mode-3 unfolding of the tensor
    % Update the following line
    diff = T3 - T3_est;
    relerr(idxiter) = sqrt(diff(:).' * diff(:)) / sqrt(T3(:).' * T3(:));
    % Check stopping criterion on relative error
    if relerr(idxiter) < th_relerr 
        break;
    end

end

% Add the relative error and the number of iterations to output structure
output.relerr = relerr;
output.numiter = idxiter;
% 
% Warning if maximum number of iterations was reached
if idxiter == options.maxiter
    warning(['The ALS algorithm reached the maximum number of ' num2str(options.maxiter) ' iterations.'])
end