function [time_used, C] = GCSC(X, k, lambda)

tic = cputime;
n = size(X, 2);
XtX = X'*X;
I = eye(n);

options = [];
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'Binary';
A = constructA(X', options);

A_tidle = I + A;
D = sum(A_tidle, 1);
D_tidle = diag(1./sqrt(D));
A_hat = D_tidle*A_tidle*D_tidle;

C = (A_hat'*XtX*A_hat + lambda*I)\(A_hat'*XtX);
time_used = cputime - tic;


