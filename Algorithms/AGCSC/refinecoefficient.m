function [Z] = refinecoefficient(C, ratio)
n = size(C, 1);
Z = zeros(n, n);

for i=1:n
    Crow = abs(C(i,:));
    sum_Crow = sum(Crow);
    [sorted, index] = sort(Crow,"descend");
    t = 0;
    j = 1;
    if ratio < 1
        while t/sum_Crow < ratio
            t = t + sorted(j);
            j = j + 1;
        end
    else
        j = ratio;
    end
    index = index(1:j);
    Z(i,index) = sorted(1:j);
end