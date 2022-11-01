function [time_used, X_bar, W] = FTRR(X, alpha)
tic = cputime;
max_iter = 1000;
k = 5;
[n,~] = size(X);  
W = eye(n);   
for t = 1:max_iter+1       
    W_old = W;
    D = diag(sum(W));
    L = eye(n)-D^(-1/2) * W * D^(-1/2);
    X_bar = X;
    for i = 1:k
        X_bar=(eye(n)-L/2)*X_bar;
    end
    S = (X_bar*X_bar'+ alpha*eye(n))\(X_bar*X_bar');   
    W = abs(S);     
    if norm(W - W_old,'fro')/norm(W_old,'fro')<1e-5
        
        break
    end
end
neighborsize = 7;
W = refinecoefficient(W, neighborsize);
   
time_used = cputime - tic;   