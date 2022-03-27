
import LogisticR.*

% i is the value of l1 regularization parameter
for i = [1e-8, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
% Load Data
load('ad_data.mat'); 

% Fit the training data
[w, c] = logistic_l1_train(X_train, y_train, i);

% Predict
res = X_test * w + c;

% Calculate AUC
[X,Y,T, AUC] = perfcurve(y_test, res, 1);

% Output
fprintf('%g, %i, %.4f\n', [i, sum(w ~= 0), AUC])
end

% Function Reference: https://github.com/jiayuzhou/CSE847/tree/master/data/alzheimers
function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations

[w, c] = LogisticR(data, labels, par, opts);
end