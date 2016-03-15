function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
n = length(y); % number of training examples
m = length(theta); %number of features
fprintf('n = %d m = %d\n', n, m);
J_history = zeros(num_iters, 1);
new_theta = zeros(m, 1); 
for iter = 1:num_iters
    for j = 1:m
        s = 0;
        for i = 1:n
            tmp = (X(i, :) * theta - y(i)) * X(i, j);
            %fprintf('tmp = %f elems = %d', tmp, numel(tmp));
            s = s + tmp; 
        end
        %fprintf('s = %f %d\n', s, numel(s));
        t = theta(j) - s * (alpha / n);
        %fprintf('t = %f %d\n', t, numel(t));
        new_theta(j) = t;
        %fprintf('done\n')
    end
    theta = new_theta;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
