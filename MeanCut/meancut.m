function [cluster] = meancut (X, varargin)
%   This is a novel spectal clustering algorithm based on path similarity and degree descent criterion. 
%   This function returns cluster labels of the N by D matrix X. Each row in X represents an observation.
% 
%   Parameters are: 
% 
%   'k1'           - A non-negative integer specifying the number of nearest neighbors for DGF. 
%                    It must be smaller than N.
%                    Default: 20
%   'k2'           - A non-negative integer specifying the number of nearest neighbors for MST. 
%                    It must be smaller than N.
%                    Default: 20
%   'ratio'        - A positive scalar in [0,1] specifying the percentile of the number of boundary 
%                    points to the total number of points. 
%                    Default: 0
%   'Embed'        - Logical scalar. If true, project X with more than 5000 samples and 50 features
%                    into a lower-dimensional space. 
%                    Default: True
%   'Normalize'    - Logical scalar. If true, scale X using min-max normalization. If features in 
%                    X are on different scales, 'Normalize' should be set to true because the clustering 
%                    process is based on nearest neighbors and features with large scales can override 
%                    the contribution of features with small scales. 
%                    Default: True
%   'Noise'        - A non-negative integer specifying the threshold of a noisy cluster.
%                    Default: 0
%   'Kernel'       - Kernel function.
%                    Default: 'Laplacian'
%   'Bandwidth'    - A positive scalar specifying the bandwidth of the kernel function.
%                    Default: 2
   
% Specify default parameters
paramNames = {'k1','k2','ratio','Embed', 'Normalize','Noise','Kernel','Bandwidth'};
defaults   = {20,20,0,true,true,0,'Laplacian',2};
[k1, k2, ratio, embed, normalize, noiseT, kernel, delta] = internal.stats.parseArgs(paramNames, defaults, varargin{:});

% Remove duplicate observations
[X, ~, orig_id] = unique(X, 'rows');

% Normalize the data
if normalize
    X = mapminmax(X',0,1)';
end

[n, dim] = size(X);
% Perform PCA for large-scale and high-dimensional data
if embed
    if(n >= 5000 && dim >= 50)
        X = init_pca(X, 50, 0.8);        
    end
end

% Calculate DGF to identify boundary and internal points 
if(ratio>0)
    [dgf_knn, dgf_dis]= knnsearch(X,X,'k',k1+1);
    dgf_knn(:,1) = [];
    dgf_dis(:,1) = [];
    density = mean(dgf_dis, 2);
    grad = sum((density(dgf_knn) - density)./dgf_dis, 2);
    grad_sort = sort(grad);
    gradT = grad_sort(ceil(n*ratio)+1);
    edg = find(grad<gradT);
    int = setdiff(1:n, edg);
else
    edg = [];
    int = 1:n;
end
Y = X(int,:);

% Construct a KNN graph
k2 = min(k2, length(int)-1);
[get_knn, knnDis]= knnsearch(Y,Y,'k',k2+1);
get_knn(:,1) = [];
knnDis(:,1) = [];
knnGraph = [sort([repelem(1:size(get_knn,1), k2)', reshape(get_knn', [], 1)],2), reshape(knnDis', [], 1)];
un_knnGraph = unique(knnGraph,'rows');
G = graph(un_knnGraph(:,1),un_knnGraph(:,2),un_knnGraph(:,3));
clear get_knn knnDis knnGraph un_knnGraph

% Generate MST
T = FastMST(Y, G, kernel, delta);

% Optimize MeanCut with degree descent
if(size(Y, 1) < 3)
    int_clust = ones(size(Y, 1) ,1);
elseif(size(Y, 1) < 50000)
    pW = path_w1(T);
    int_clust = degree_descent1(pW,noiseT);
else
    degs = path_w(T);
    int_clust = degree_descent(T,degs,noiseT);
end

clust = zeros(n,1);
clust(int) = int_clust;
near_int = knnsearch(X(int,:),X(edg,:));
clust(edg) = int_clust(near_int);
cluster = clust(orig_id,:);

end
