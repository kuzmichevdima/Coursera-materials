function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%fprintf('size of X: %d %d\n', size(X));
%h_theta = X * theta(2:end, :) + theta(1);
h_theta = X * theta;
%fprintf('size of theta: %d %d size of y: %d %d size of h_theta: %d %d\n', size(theta), size(y), size(h_theta));
reg_term = lambda / (2 * m) * sum(theta(2 : end) .^ 2); 
assert(size(h_theta) == size(y));
J = (1 / (2 * m)) * sum((h_theta - y) .^ 2) .+ reg_term;
grad = zeros(size(theta));
n = columns(X);
for j = 1:n
    grad(j) = (1 / m) * sum((h_theta - y) .* X(:, j));
    if j != 1
        grad(j) += (lambda / m) * theta(j);
    endif
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
