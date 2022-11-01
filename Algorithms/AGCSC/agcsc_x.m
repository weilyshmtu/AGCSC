function [F, C, time_used] = agcsc_x(X, alpha, beta)

% ||F - SX||_F^2 + alpha * ||X - CF||_F^2 + beta||C - SC||_F^2
% s.t. Ce = e, C = C', C>0, S = 0.5*(C + I)

% X is a data matrix with size n x d, n is the number of sample, d is the
% original dimensionality, C is the reconstruction coefficient matrix




t0 = cputime;
%% parameters
tol = 1e-7;
maxIter = 1e4;
rho = 1.1;
max_mu = 1e30;
mu = 1e-6;

[n,~] = size(X);
e = ones(n,1);
I = eye(n);

XXt = X*X' ;
eet = e*e';

%% auxiliary Variables
F = X;
Z = zeros(n, n);


%% Lagrange multipliers
Y1 = zeros(n, n);
Y2 = zeros(n, 1);

%% Start main loop
iter  = 0;
while iter < maxIter
    iter = iter + 1;

    % update C
    FFt = F*F';
    I_Z = I - Z;
    A = 2*XXt + 2*alpha*FFt + 2*beta*(I_Z*I_Z') + mu*(I + eet);
    B = 4*F*X' - 2*XXt + 2*alpha*(X*F') + mu*(Z + eet) - Y1 - Y2*e';
    C = B/A;

    % update F
    CtC = C'*C;
    A = alpha*CtC + 2*I;
    B = C*X + X + alpha * C'*X;
    F = A\B;

    % update Z
    A = 2*beta*CtC + mu*I;
    B = 2*beta*CtC + Y1 + mu*C;
    Z = A\B;
    Z = Z - diag(diag(Z));
    Z = 0.5*(Z + Z');
    Z = max(Z, 0);
    

    leq1 = C - Z;
    leq2 = C*e - e;

    stopC = max([max(max(abs(leq1))),max(max(abs(leq2)))]);
    
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',leq1=' num2str(max(max(abs(leq1))),'%2.3e')  ...
            ',leq2=' num2str(max(max(abs(leq2))),'%2.3e')]);
    end
    if stopC<tol 
        time_used = cputime - t0;
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end

end

