clc;
best_ac = max(max(accuracy));
[row col]= find(best_ac==accuracy);
best_ac_lambda = par.lambda(row);
best_ac_adjk = par.adjKnn(col);
best_ac_nmi = nmi(row,col);

fprintf(['The best accuracy is ' num2str(best_ac)  ', the corresponding nmi is ' num2str(best_ac_nmi(1)), ...
     ', \n time = ' num2str(time(row(1),col(1)))...
     ', \n when lambda = ' num2str(best_ac_lambda) ', and adjk = ' num2str(best_ac_adjk) '\n']);

best_nmi = max(max(nmi));
[row col]= find(best_nmi==nmi);
best_nmi_lambda = par.lambda(row);
best_nmi_adjk = par.adjKnn(col);
best_nmi_ac = accuracy(row,col);
fprintf(['The best NMI is ' num2str(best_nmi)  ', the corresponding nmi is ' num2str(diag(best_nmi_ac)'), ...
    '\n, when lambda = ' num2str(best_nmi_lambda) ', and adjk = ' num2str(best_nmi_adjk) '\n']);