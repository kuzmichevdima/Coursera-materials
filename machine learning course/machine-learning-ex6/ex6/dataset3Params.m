function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

%val_set = [0.01 0.03 0.1 0.3 1 3 10 30];
%best = 1e9;
%best_C = -1;
%best_sigma = -1;
%for i = 1:8
    %for j = 1:8
        %C = val_set(i);
        %sigma = val_set(j);
        %model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        %predictions = svmPredict(model, Xval);
        %pred_error = mean(double(predictions - yval));
        %cnt = 0;
        %for h = 1:size(yval)
            %if yval(h) != predictions(h)
            %    cnt += 1;
            %endif
        %end
        %fprintf('C = %f sigma = %f cnt = %d\n', C, sigma, cnt);
        %if cnt < best
            %best = cnt;
            %best_C = C;
            %best_sigma = sigma;
        %endif
    %end
%end
%fprintf('best = %d best_C = %f best_sigma = %f\n', best, best_C, best_sigma);


C = 0.3;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
