function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
for i = 1:m
    if y(i) == 1
        J = J + -log(sigmoid(X(i, :) * theta));
    else
        J = J - log(1 - sigmoid(X(i, :) * theta));
    endif
end
J /= m;
% disp(size(J));
grad = zeros(size(theta));
n = columns(X);
% X is mxn matrix, theta is nx1, y is mx1. m is number of examples, n is the number of features
% fprintf('dimensions if X : %d %d n = %d m = %d\n', rows(X), columns(X), n, m);
 %fprintf('dimensions of theta: %d %d\n', rows(theta), columns(theta));
 %disp(size(theta));
 %disp(size(X * theta));
 %disp(size(sigmoid(X * theta)));
for j = 1:n
    % (sigmoid(X * theta) - y) is mx1, transposed is 1xm, X(:, j) is mx1
    %fprintf('sigmoid: %d %d\n', rows(sigmoid(X * theta)), columns(sigmoid(X * theta)));
    grad(j) = (1 / m) * (sigmoid(X * theta) - y)' * X(:, j);
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
