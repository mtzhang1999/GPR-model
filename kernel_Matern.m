function [K, dis] = kernel_Matern(t,l)
    % Squared Exponential Kernel
    % input: time vector, sigma1, l1, sigma2, l2;
    % l2 should be approximately 10 times of l1
    % because it is used to learn for long-time trend 
    % output: K matrix, distance matrix(squared)
    len = length(t);
    T = t';
    temp1 = T(ones(len, 1), :);
    temp2 = t(:, ones(1, len));
    dis = (abs(temp1-temp2));
    K = (1+sqrt(3)*dis/l).*exp(-sqrt(3)*dis/l);
end