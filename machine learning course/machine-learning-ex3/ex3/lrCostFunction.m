function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = rows(X); % number of training examples
n = size(theta);
grad = zeros(n);
% You need to return the following variables correctly 
% y is mx1, theta is nx1
%fprintf('sizes of X:\n');
%disp(size(X));
%fprintf('sizes of y:\n');
%disp(size(y));
%fprintf('sizes of theta:\n');
%disp(size(theta));
htheta = sigmoid(X * theta);
%fprintf('sizes of htheta:\n');
%disp(size(htheta));
%disp(htheta);
%log1 and log2 are mx1
log1 = log(htheta);
log2 = log(1 - htheta);
%fprintf('log1 sizes\n');
%disp(size(log1));
theta0 = theta;
theta0(1) = 0;
J = (1 / m ) * sum(-y .* log1 .- (1 - y) .* log2) + (lambda / (2 * m)) * sum(theta0 .^ 2); 
grad = (1 / m) * X' * (htheta - y) + (lambda / m) * theta0;
%fprintf('sizes of grad:\n');
%disp(size(grad));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end
