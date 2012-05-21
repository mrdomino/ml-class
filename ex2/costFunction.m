function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
h = sigmoid(X * theta); % Vector of hypothesis values

J = 1 / m * (-y' * log(h) - (1 .- y') * log(1 .- h));

grad = 1 / m * (h .- y)' * X;

end
