function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h = sigmoid(X * theta); % Vector of hypothesis values

J = (1 / m * (-y' * log(h) - (1 .- y') * log(1 .- h))
     + lambda / (2 * m) * theta(2:end)' * theta(2:end));

grad = 1 / m * X' * (h .- y);

grad(2:end) = grad(2:end) + theta(2:end) * lambda / m;

end
