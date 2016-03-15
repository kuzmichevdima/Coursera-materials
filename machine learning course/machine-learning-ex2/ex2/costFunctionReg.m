function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
for i = 1:m
    if y(i) == 1
        J = J - log(sigmoid(X(i, :) * theta));
    else
        J = J - log(1 - sigmoid(X(i, :) * theta));
    endif
end
%disp(size(J));
%disp(J);
n = columns(X);
J = J / m + lambda / (2 * m) * sum(theta(2:n) .^ 2);
grad = zeros(size(theta));
for j = 1:n
    grad(j) = (1 / m) * (sigmoid(X * theta) - y)' * X(:, j);
    if j != 1
        grad(j) = grad(j) + (lambda / m) * theta(j);
    endif
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
