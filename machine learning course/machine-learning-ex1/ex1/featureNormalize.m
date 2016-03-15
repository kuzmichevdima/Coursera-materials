function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
M = size(X, 2);
mu = zeros(1, M);
sigma = zeros(1, M);
N = rows(X);

fprintf('N = %d M = %d\n', N, M);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
for j = 1:M
    mu(j) = mean(X_norm(:, j));
    for i = 1:N
        X_norm(i, j) -= mu(j);
    end
end
for j = 1:M
    sigma(j) = std(X_norm(:, j));
    for i = 1:N
        X_norm(i, j) /= sigma(j);
    end
end









% ============================================================

end
