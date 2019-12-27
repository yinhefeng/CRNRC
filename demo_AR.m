clear
clc
close all

% data loading (here we use the AR dataset as an example)
load('AR_DAT.mat');

% -------------------------------------------------------------------------
% parameter setting
par.nClass = length(unique(trainlabels)); % the number of classes in the subset of AR database
dim = [54 120 300];
lambda = [1e-3,1e-2,1e-3];
mu= 1e-1;

%--------------------------------------------------------------------------
Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   trainlabels(trainlabels<=par.nClass);
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   testlabels(testlabels<=par.nClass);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels

train_tol= size(Tr_DAT,2);
test_tol = size(Tt_DAT,2);
ClassNum = par.nClass;
%--------------------------------------------------------------------------
reg_rate = zeros(1,length(dim));
kk = 1;

param = [];
% param.lambda = lambda;
param.mu = mu;

for eigen_num=dim
    %eigenface extracting
    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,eigen_num);
    tr_dat  =  disc_set'*Tr_DAT;
    tt_dat  =  disc_set'*Tt_DAT;
    
    % normalize to unit L2 norm
    tr_dat = normc(tr_dat);
    tt_dat = normc(tt_dat);
    % tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [par.nDim,1]) );
    % tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [par.nDim,1]) );
    
    % pre-calculation, reduce the running time
    class_num = length(unique(trls));
    tr_sym_mat = zeros(length(trls));
    for ci = 1 : class_num
        ind_ci = find(trls == ci);
        tr_descr_bar = zeros(size(tr_dat));
        tr_descr_bar(:,ind_ci) = tr_dat(:, ind_ci);
        tr_sym_mat = tr_sym_mat + lambda(kk) * (tr_descr_bar' * tr_descr_bar);
    end
    
    XTX = tr_dat'*tr_dat;
    temp_X = (XTX+tr_sym_mat+mu/2*eye(train_tol))\eye(train_tol);
    
    ID = zeros(1,test_tol);
    X = tr_dat;
    param.lambda = lambda(kk);
    for i=1:test_tol
        y = tt_dat(:,i);
        
        % obtain the coding vector
        [z,c] = CRNRC(X, temp_X, y, param);
        
        W = sparse([],[],[],train_tol,ClassNum,length(c));
        
        for j=1:ClassNum
            ind = (j==trls);
            W(ind,j) = c(ind);
        end
        
        % compute the classwise residual
        temp = X*W-repmat(y,1,ClassNum);
        residual = sqrt(sum(temp.^2));
        
        % classification
        [~,index]=min(residual);
        ID(i)=index;
    end
    
    %-------------------------------------------------------------------------
    cornum      =   sum(ID==ttls);
    reg_rate(kk)         =   cornum/length(ttls); % recognition rate
    kk = kk+1;
end

% output the result
disp([dim;reg_rate])