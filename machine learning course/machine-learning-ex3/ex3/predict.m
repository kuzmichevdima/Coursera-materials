function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
n = size(X, 2);
%fprintf('X size:\n');
%disp(size(X));
%fprintf('Theta1 size:\n');
%disp(size(Theta1));
%fprintf('Theta2 size:\n');
%disp(size(Theta2));
%X is mx1, Theta1 is 
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
for query = 1:m
    x = X(query, :);
    x = [1; x'];
    %fprintf('x:\n');
    %disp(x);
    %x is nx1, Theta1 is ?xn
    z2 = Theta1 * x;
    a2 = [1; sigmoid(z2)];
    %disp(a2);
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    best_prob = 0.0;
%    fprintf('num_labels = %d a3 size:\n', num_labels);
    %disp(size(a3));
    for digit = 1 : num_labels
        digit_prob = a3(digit);
        if (digit_prob > best_prob)
            best_prob = digit_prob;
            p(query) = digit;
        endif
    end
%    fprintf('best_prob = %f\n', best_prob);
end

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
