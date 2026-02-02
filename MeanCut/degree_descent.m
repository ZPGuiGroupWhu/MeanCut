function [cluster] = degree_descent(T,degs,noiseT)
%% Greedy optimization with degree descent criterion algorithm

n = size(T.Nodes,1);
cluster = zeros(n,1);
mark = 1;
%% Arrange all points in degree descending order
[deg_sort, sort_id] = sort(degs,'descend');
while ismember(0,cluster)
    id = find(cluster==0);
    num = 1;
    sum_w = 1;
    sum_deg = deg_sort(id(1));
    meancut = (1/(n-1))*(1-sum_w/sum_deg);
    storage = sort_id(id(1));
    [~,~,e_id] = shortestpathtree(T,sort_id(id(1)),sort_id(id(2:end)),'OutputForm','cell');
    e_val = cellfun(@(x) T.Edges.Weight(x), e_id, 'UniformOutput', false);
    pw_val = cell2mat(cellfun(@min, e_val, 'UniformOutput', false));
    for i=2:length(id)
        num = num + 1;
        sum_deg = sum_deg + deg_sort(id(i));
        sum_w = sum_w + 2*pw_val(i-1)*length(storage)+1;
        cut =  (1/(n-num))*(1-sum_w/sum_deg);
        if(cut <= meancut)
            meancut = cut;
            storage = [storage;sort_id(id(i))];
        else
            num = num - 1;
            sum_deg = sum_deg - deg_sort(id(i));
            sum_w = sum_w - 2*pw_val(i-1)*length(storage)-1;
        end
    end
    cluster(ismember(sort_id,storage)) = mark;
    mark = mark + 1;
end
temp = sortrows([sort_id,cluster]);
cluster = temp(:,2);
mark = 1;
for i=1:max(cluster)
    num = length(find(cluster==i));
    if(num < noiseT)
        cluster(cluster==i) = 0;
    else
        cluster(cluster==i) = mark;
        mark = mark + 1;
    end
end

end
