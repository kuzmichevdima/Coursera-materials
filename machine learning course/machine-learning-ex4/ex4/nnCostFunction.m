function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
%fprintf('m = %d n = %d\n', m, n);
         
% You need to return the following variables correctly 
J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));
K = num_labels;
H = hidden_layer_size;
delta_1 = lambda * [zeros(H, 1) Theta1(:, 2:end)];
delta_2 = lambda * [zeros(K, 1) Theta2(:, 2:end)];
%assert(size(delta_2) == size(Theta2));
for i = 1:m
    a1 = X(i, :)';
    a1 = [1; a1];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
%    assert(size(a2, 1) == H + 1);
 %   assert(size(a2, 2) == 1);
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
  %  assert(size(a3, 1) == K);
   % assert(size(a3, 2) == 1);
    y_vect = zeros(K, 1);
    if y(i) == 0
        y_vect(K) = 1;
    else
        y_vect(y(i)) = 1;
    endif
    for k = 1:K
        if y_vect(k) == 1
            J = J + log(a3(k));
        else
            J = J + log(1 - a3(k));
        endif
    end

    sigma_3 = a3 - y_vect;
    %assert(size(sigma_3, 1) == K);
    %assert(size(sigma_3, 2) == 1);
%    fprintf('size sigma_3\n');
%    disp(size(sigma_3));
    sigma_2 = Theta2' * sigma_3 .* (a2 .* (1 - a2));
%    assert(size(sigma_2, 1) == 1);
 %   assert(size(sigma_2, 2) == hidden_layer_size + 1);
    %sigma_1 = sigma_2(1, 2:end) * Theta1 .* (a1 .* (1 - a1));
    %assert(size(sigma_1, 1) == 1);
    %assert(size(sigma_1, 2) == n + 1);
%    assert(size(delta_2) == size(Theta2));
    delta_2 = delta_2 + sigma_3 * a2';
    delta_1 = delta_1 + sigma_2(2:end) * a1';
end
J = J * (-1.0 / m);
%adding regularization term
reg_term = 0;
Theta1_squared = Theta1 .**2;
Theta2_squared = Theta2 .**2;
summand_1 = sum(sum(Theta1_squared(:, 2:end)));
%disp(size(summand_1));
%fprintf('summand_1 = %.5f\n', summand_1);
summand_2 = sum(sum(Theta2_squared(:, 2:end)));
%fprintf('summand_2 = %.5f\n', summand_2);
reg_term += summand_1;
reg_term += summand_2;
%for i = 1:hidden_layer_size
%    for j = 2:(n + 1)
%        reg_term += Theta1(i, j)**2;
%    end
%end
%for i = 1:K
%    for j = 2:(hidden_layer_size + 1)
%        reg_term += Theta2(i, j)**2; 
%    end
%end

reg_term = reg_term * (lambda / (2 * m));
J += reg_term;
%disp(J);
%fprintf('delta_1:\n');
%disp(delta_1);
%fprintf('delta_2:\n');
%disp(delta_2);
delta_1 = delta_1 .* (1 / m);
%fprintf('size delta_2:\n');
%disp(size(delta_2));
delta_2 = delta_2 .* (1 / m);
%disp(delta_1);
%fprintf('size delta_2:\n');
%disp(size(delta_2));
%fprintf('size Theta2\n');
%disp(size(Theta2));
%assert(size(delta_1) == size(Theta1));
%assert(size(delta_2) == size(Theta2));
grad = [delta_1(:); delta_2(:)];
%fprintf('in cost function');
%disp(grad);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients


end
