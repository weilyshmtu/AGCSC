
%% Image clustering Test
clear, clc

%% Experiment Settings
addpath(genpath(pwd));
% load data
database = {'ORL','YALEB','COIL20','COIL40','Umist','MNIST'};
numdatas = length(database);

ProjectionType = 0;          % data projection type    
NormalizationType = 2;         % data normalization type               

for dataindex = 2:numdatas   % different databases
    DataName = database{dataindex};
    load(database{dataindex});

    X = fea;
    L = gnd;

    if min(unique(L)) == 0
        L = L + 1;
    end
    nbcluster = max(unique(L));
   

    % projection
    switch ProjectionType
        case 0
            X = X;
        case 1
            X =  DataProjection(X, dim);
    end
 
    % normalization
    switch NormalizationType
        case 0
            X = X;
        case 1
            X = mexNormalize(X);
        case 2 
            if max(max(X)) > 1
                X = X./repmat((255)*ones(1,size(X,2)),size(X,1),1);
            end
    end

    % parameters 
    %alpha = [1e-6, 1e-5, 0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05, ...
    %    0.08, 0.1, 0.5];
    %beta = [1e-6, 1e-5, 0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05, ...
    %    0.08, 0.1, 0.5];
    alpha = [1e-5,1e-4,1e-3,5e-3,0.01,0.05,0.1,0.5];
    beta  = [1e-5,1e-4,1e-3,5e-3,0.01,0.05,0.1,0.5];
    % algorithm
    Timecell = cell(length(alpha), length(beta));
    for a=1:length(alpha)
        for b=1:length(beta)
            [F, C, time_used] =  agcsc_x(X, alpha(a), beta(b)); 
            C1 = C;
            [idx,~] = clu_ncut(C1,nbcluster);
            acc_array1(a, b) = compacc(idx',L)
            nmi_array1(a, b) = nmi(L, idx')
            Timecell{a,b} = time_used;
          
            % save features and coefficient matrices
            tdir ="./Results/" + database{dataindex};
            cd(tdir)
            tfilename = num2str(a) + "_" + num2str(b) ;
            save(tfilename, "F", "C")  
            cd ..
            cd ..
        end
    end
    % save accuracy and nmi
    filename = DataName + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
    save(filename, "acc_array1", "nmi_array1", "Timecell")
end