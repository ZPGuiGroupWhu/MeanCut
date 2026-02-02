![image](https://img.shields.io/badge/MATLAB-R2023a-red)
# MeanCut: Greedy graph clustering by fast maximum spanning tree and degree descent criterion

We propose a novel graph clustering algorithm. Our key contribution lies in developing an single-cluster-oriented graph cut function MeanCut and leveraging the degree descent criterion to greedily optimize MeanCut. It is theoretically proved that the degree descent criterion guarantees the global optimality under the premise of path-based similarity and predefined assumptions. Credit to the density gradient factor (DGF), MeanCut enables the effective separation of weakly connected clusters. A fast approximate maximum spanning tree (FastMST) improves the time efficiency of MST generation using a KNN-based sparse graph. The experimental results demonstrated that MeanCut presents distinct advantages in clustering accuracy, parameter robustness, and time scalability and exhibits remarkable potentials for downstream tasks.

![image](https://github.com/ZPGuiGroupWhu/MeanCut-Clustering/blob/main/Pic/github.png)

# How To Run

Download the code and run the 'main' file in the 'MeanCut' file. ***To be noted, the Deep Learning Toolbox should be installed in MATLAB before running the application of Face Recognition.***

The 'meancut' function provides multiple hyperparameters for user configuration as follows 
```matlab
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
%   'ratio'        - A positive scalar in [0,1] specifying the percentile of the number of boudanry 
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
```

The 'main.m' file provides an example
```matlab
% Input the data
clear;
data = textread('Synthetic Datasets\DS1.txt');
[~, m] = size(data);
X = data(:, 1:m-1);
ref = data(:, m);

% Perform the MeanCut algorithm
% DS6, DS7, DS8: k1 = 20; ratio=0.2, 0.7, 0.3;
cluster = meancut(X);

% Evaluate the clustering accuracy
addpath ClusterEvaluation
[ACC, NMI, ARI, Fscore, JI, RI] = ClustEval(ref, cluster);

% Visualize the clustering results
plotcluster2(X, cluster);
```

# License

This project is covered under the Apache 2.0 License.
