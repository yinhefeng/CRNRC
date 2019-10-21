function [z,c] = CRNRC(X,temp_X, y, param)
lambda = param.lambda;
mu = param.mu;
[~,n] = size(X);
tol = 1e-5;
maxIter = 5;
% mu= 1e-1;
z = zeros(n,1);
c = zeros(n,1);
delta = zeros(n,1);

% XTX = X'*X;
XTy = X'*y;
iter = 0;

% class_num = length(unique(tr_label));
% tr_sym_mat = zeros(n);
% for ci = 1 : class_num
%     ind_ci = find(tr_label == ci);
%     tr_descr_bar = zeros(size(X));
%     tr_descr_bar(:,ind_ci) = X(:, ind_ci);
%     tr_sym_mat = tr_sym_mat + lambda * (tr_descr_bar' * tr_descr_bar);
% end

% temp_X = pinv(XTX+tr_sym_mat+mu/2*eye(n));
% temp_X = inverse(XTX+tr_sym_mat+mu/2*eye(n));

while iter<maxIter
    iter = iter + 1;
    
    zk = z;
    ck = c;
    
    % update c
    %     c = (XTX+tr_sym_mat+mu/2*eye(n))\((1+lambda)*XTy+mu/2*z+delta/2);
    c = temp_X*((1+lambda)*XTy+mu/2*z+delta/2);
    
    % update z
    z_temp = c-delta/mu;
    z = max(0,z_temp);
    
    leq1 = z-c;
    leq2 = z-zk;
    leq3 = c-ck;
    stopC1 = max(norm(leq1),norm(leq2));
    stopC = max(stopC1,norm(leq3));
    %     disp(stopC)
    
    if stopC<tol || iter>=maxIter
        break;
    else
        delta = delta + mu*leq1;
    end
end