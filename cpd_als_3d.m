function [B1, B2, B3, c, output] = cpd_als_3d(T, R, options, varargin)
% CPD_ALS_3D Computes the canonical polyadic decomposition (CPD) of a
% 3rd-order tensor using the alternative least squares algorithm (ALS) 
% (assignment solution).
%
%INPUT:
%   T (I_1 x I_2 x I_3): 3-D tensor data tensor
%   R (1 x 1): number of CPD components
%   options (struct, optional) : optimization options containing:
%          - th_relerr (1 x 1): relative error threshold
%          - maxiter   (1 x 1): max number of iterations
%   variable extra inputs (1 x 3 cell array OR 3 arrays, OPTIONAL): 
%           initialization for the factor matrices
%
%OUTPUT:
%   B1 (I_1 x R): mode-1 factor matrix with normalized columns 
%   B2 (I_2 x R): mode-2 factor matrix with normalized columns 
%   B3 (I_3 x R): mode-3 factor matrix with normalized columns
%   c (1 x R):  vector containing the respective weights for the CPD
%               components
%   output (struct) : optimization options containing:
%          - numiter (1 x 1): the number of iterations the algorithm ran for
%          - relerr (1 x numiter): relative error achieved, defined as Frobenius norm of 
%           the residual of the decomposition OVER Frobenius norm of the 
%           original tensor.
%
% Remarks: 
%   The order in which the factor matrices is updated is fixed: mode-1, 
%   then mode-2 and, lastly, mode-3. 
%   
% Authors: Borbala Hunyadi (b.hunyadi@tudelft.nl)
%          Sofia-Eirini Kotti (s.e.kotti@tudelft.nl)

% Get algorithm parameters
% Number of iterations
maxiter = options.maxiter;
% Threshold for relative error
th_relerr = options.th_relerr;
%
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
    B1 = randn(size(T, 1), R);
    B2 = randn(size(T, 2), R);
    B3 = randn(size(T, 3), R);
else                % Otherwise use the given initialization
    assert(all(cellfun(@(x) size(x,2), init) == R), ...
        "The given initialization has a different number of rank-1 components than the given R.")
    B1 = init{1};
    B2 = init{2};
    B3 = init{3};
end

% ====================== YOUR CODE HERE ======================
% You need to calculate the following variables correctly (you should comment 
% the following 5 lines out)
% B1 = 0;
% B2 = 0;
% B3 = 0;
% c = 1;
% relerr = Inf;

% Normalize the columns of the initial factor matrices
B1 = B1 ./ vecnorm(B1);
B2 = B2 ./ vecnorm(B2);
B3 = B3 ./ vecnorm(B3);

% Obtain the three tensor unfoldings 
T1 = hidden_mode_n_matricization(T,1);
T2 = hidden_mode_n_matricization(T,2);
T3 = hidden_mode_n_matricization(T,3);

% ALS iterations
for idxiter = 1:maxiter

    % Mode 1 (do not forget to normalize the columns!)
    B1 = T1 * hidden_khatri_rao(B3, B2) * pinv((B3.'*B3) .* (B2.'*B2));
    B1 = B1 ./ vecnorm(B1);
    % Mode 2 (do not forget to normalize the columns!)
    B2 = T2 * hidden_khatri_rao(B3, B1) * pinv((B3.'*B3) .* (B1.'*B1));
    B2 = B2 ./ vecnorm(B2);
    % Mode 3 (do not forget to normalize the columns!)
    B3 = T3 * hidden_khatri_rao(B2, B1) * pinv((B2.'*B2) .* (B1.'*B1));
    c = vecnorm(B3);
    B3 = B3 ./ c;
    % Compute the current estimate of the mode-3 unfolding of the tensor
    T3_est = B3 * diag(c) * (hidden_khatri_rao(B2, B1)).';

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
