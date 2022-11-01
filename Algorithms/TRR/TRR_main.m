% =========================================================================
% An example code for the algorithm proposed in
%
%   Xi Peng, Zhang Yi, and Huajin Tang.
%   Robust Subspace Clustering via Thresholding Ridge Regression.
%   The Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI), Austin, Texas, USA, January 25–29, 2015.
%
% Written by Xi Peng @ I2R A*STAR
% Nov., 2014.
% =========================================================================



close all;
clear all;
clc;

%% --------------------------------------------------------------------------
addpath ('../usages/');
addpath ('../data/');

fprintf('Beginning!\n');
% loading data

% % ==================================  
CurData = 'ExYaleB_54_48_permute';
load (CurData);  
% parameters configuration
% par.nClass             =   39;% the first nClass subjects are used to test
% par.nDim               =   116;% feature dimensionality of PCA
% par.lambda             =   [1.0];
% par.adjKnn             =   [5];

% % ==================================  
CurData = 'AR_55_40_permute';
load (CurData);  
% parameters configuration
par.nClass             =   100;% the first nClass subjects are used to test
par.nDim               =   167;% feature dimensionality of PCA
par.lambda             =   [0.7];
par.adjKnn             =   [7];

%% --------------------
% each column of DAT denotes a data point
DATA       =   double(DAT(:,labels<=par.nClass));
labels     =   labels(labels<=par.nClass);

dat = FeatureEx(DATA, par);
clear DATA;

% --- get the clustering result based on L2-graph
[accuracy nmi time Time_BuildGraph] = L2_code_vector(dat, labels, par);
clear tmp1 tmp2 tmp3 i DAT1 DAT2 LABEL1 LABEL2 dat1 dat2 dat order;
clear DAT DATA ans labels;
save (['TRR_' CurData '_nClass' num2str(par.nClass) '_nDim' num2str(par.nDim) '_nlambda' num2str(length(par.lambda)) '_knn' num2str(length(par.adjKnn))])
