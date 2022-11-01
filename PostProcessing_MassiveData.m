% To obtain the results of TAGCSC

clear
clc
addpath '.\Databases\Face Image Databases'
addpath '.\Databases\MNIST'
addpath '.\Databases\COIL'
addpath '.\Measurements'
addpath '.\Algorithms'

database = {'ORL','AR','COIL20','COIL40','Umist','MNIST'};

para_rows = 8;
para_cols = 8;

for dataindex = 2:2%length(database)

    % load data
    load(database{dataindex});
    X = fea;
    L = gnd;
    nbcluster = max(unique(L));
    
    NeighborSize = 4:15;
    post_acc_results = cell(1, length(NeighborSize));
    post_nmi_results = cell(1, length(NeighborSize));

    acc_array = 0;
    nmi_array = 0;

    tdir = ".\Results\" + database{dataindex};
    cd(tdir)
    
    for ri = 1:length(NeighborSize)
        acc_array = min(acc_array, 0);
        nmi_array = min(nmi_array, 0);
        for r=1:para_rows
            for c=1:para_cols
                tfilename = num2str(r) + "_" + num2str(c);
                load(tfilename)
                tC = C;
                [Z] = refinecoefficient(tC, NeighborSize(ri));
    
                [idx,~] = clu_ncut(Z,nbcluster);
                acc_array(r, c) = compacc(idx',L)
                nmi_array(r, c) = nmi(L, idx')
            end
        end
        post_acc_results{ri} = acc_array
        post_nmi_results{ri} = nmi_array
    end
    for j=1:length(post_acc_results)
        max_acc_array(j) = max(max(post_acc_results{j}));
        max_nmi_array(j) = max(max(post_nmi_results{j}));
    end
    max_acc_array
    max_nmi_array
%     
    cd ..
    filename = "Post_Processing_" +database{dataindex};
    save(filename, "post_acc_results", "post_nmi_results", "max_acc_array", "max_nmi_array")

end


