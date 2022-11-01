
clear, clc

%% Experiment Settings
% add necessary folds
addpath(genpath(pwd));

% load data
database = {'ORL','YALEB','COIL20','COIL40','Umist','MNIST'};
numdatas = length(database);
algorithms = {'SSC', 'LRR', 'LRSC', 'LSR', 'BDR', 'TRR', 'FLSR', 'FTRR', 'GCSC'};
numalgs = length(algorithms);

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

    for algindex = 1:numalgs
        algname = algorithms{algindex};
        acc_array1 = 0; nmi_array1 = 0;
        switch algname
            %% SSC
            case 'SSC'
                disp('Running algorithm is SSC!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = ssc(X, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..
            
            %% LRR    
            case 'LRR'
                disp('Running algorithm is LRR!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                 [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C, ~] = lrr(X, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..
            
             %% LRSC
             case 'LRSC'
                disp('Running algorithm is LRSC!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                 [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = lrsc_noiseless(X, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..
            
            %% LSR    
            case 'LSR'
                disp('Running algorithm is LSR!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = LSR1(X, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..
            
            case 'BDR'
                disp('Running algorithm is BDR!')
                % parameters 
                alpha = [0.1,1,10,20,50,80];
                beta = [0.001,0.01,0.1,0.5,1,5,10,20,50];
           
                Timecell = cell(length(alpha), length(beta));
                [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                for a=1:length(alpha)
                    for b=1:length(beta)
                        [time_used,~,C] = BDR_solver(X, nbcluster, alpha(a), beta(b));
                        C1 = C;
                        [idx,~] = clu_ncut(C1,nbcluster);
                        acc_array1(a, b) = compacc(idx',L)
                        nmi_array1(a, b) = nmi(L, idx')
                        Timecell{a,b} = time_used;
                    end
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..

            %% TRR
            case 'TRR'
                disp('Running algorithm is TRR!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                ratio = 7;
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = LSR1(X, lambda(para_index));
                    for ratioindex=1:1%length(ratio)
                        [C1] = refinecoefficient(C, ratio);
                        [idx,~] = clu_ncut(C1,nbcluster);
                        t_acc_array1(ratioindex) = compacc(idx',L);
                        t_nmi_array1(ratioindex) = nmi(L, idx');
                    end
                    acc_array1(para_index) = max(t_acc_array1)
                    nmi_array1(para_index) = max(t_nmi_array1)
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..

             case 'FLSR'
                disp('Running algorithm is FLSR!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                [m1, m2] = size(X);
                if m2 == length(L)
                    X = X';
                end
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = FLSR(X, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..

            case 'FTRR'
                disp('Running algorithm is FTRR!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                [m1, m2] = size(X);
                if m2 == length(L)
                    X = X';
                end
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = FTRR(X, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..
             
             
            case 'GCSC'
                disp('Running algorithm is GCSC!')
                % parameters 
                lambda = [1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 80, 100];
                Timecell = cell(length(lambda));
                [m1, m2] = size(X);
                if m1 == length(L)
                    X = X';
                end
                neighborsize = 7;
                for para_index = 1:length(lambda)
                    [Timecell{para_index}, C] = GCSC(X, neighborsize, lambda(para_index));
                    C1 = C;
                    [idx,~] = clu_ncut(C1,nbcluster);
                    acc_array1(para_index) = compacc(idx',L)
                    nmi_array1(para_index) = nmi(L, idx')
                end
                tdir ="./Results/";
                cd(tdir)
                filename = DataName + "_" + algname + "_" + num2str(ProjectionType)+ "_"+ num2str(NormalizationType);
                save(filename, "acc_array1", "nmi_array1", "Timecell")
                cd ..
              
        end
    end
    
end