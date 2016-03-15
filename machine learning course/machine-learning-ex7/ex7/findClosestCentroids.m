function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
% You need to return the following variables correctly.
m = rows(X);

%fprintf('sizes of centroids: %d %d m = %d features = %d', size(centroids), m, columns(X))
idx = zeros(m, 1);
for example = 1:m
    for c = 1:K
        dist = sum((X(example, :) - centroids(c, :)) .^ 2);
        if c == 1 || dist < best
            chosen = c;
            best = dist;
        end
    end
    idx(example) = chosen;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

