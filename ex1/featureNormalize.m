function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
sigma = std(X);
for i = 1:size(X, 1),
    for j = 1:size(X, 2),
        X_norm(i, j) = (X(i, j) - sigma(j)) / mu(j);
    end
end




% ============================================================

end
