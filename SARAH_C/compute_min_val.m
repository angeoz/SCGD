function [out] = compute_min_val(data, config)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[m n] = size(data);
data_mean = mean(data, 1);
%data_mid = data - data_mean;
A = zeros(n);
for i = 1:m
    A = A + (data(i,:)-data_mean)' * (data(i,:) - data_mean);
end
A = 2 * A/m;
b = mean(data, 1);
x = linsolve(A, b');
out = mean((data * x - mean(data * x)).^2) - mean(data * x);

end

