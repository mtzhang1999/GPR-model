close all; clear all; clc;
file_name = 'ORCL-Max.csv';
total_len = 200;
inverse = 1;
[raw_data, ~] = Data_Preprocess(file_name, total_len, inverse);
len = length(raw_data);
t = (1:len)';
train_len = 165;
validation_len = 25;
data_ave = mean(raw_data(1:train_len));
y = raw_data - data_ave;
test_len = total_len - train_len - validation_len;
train_y = y(1:train_len);
train_t = t(1:train_len);
val_y = y(train_len + 1:train_len + validation_len);
val_t = t(train_len + 1:train_len + validation_len);
%% PSO 
% PSO constant definition
w = 0.6;
c1 = 2;
c2 = 2;
% Particle Matrix Length: 13
% [Sigma1, l1, Sigma2, l2]
% Current Position:4 
% Velocity:4
% Personal Best:4 
% Best MSE:1
sigma1_interval = [0, 10];
sigma2_interval = [20, 40];
l1_interval = [0, 2];
l2_interval = [10, 40];
Global_Best = ones(1, 5) * 3e8;
iter_Max = 50;
MSE_plot = zeros(1, iter_Max);
v_max = 5;
% Random Initialization
Rand_Sigma1 = rand([50, 1]) * sigma1_interval(2);
Rand_Sigma2 = rand([50, 1]) * (sigma2_interval(2) - sigma2_interval(1)) + sigma2_interval(1);
Rand_l1 = rand([50, 1]) * l1_interval(2);
Rand_l2 = rand([50, 1]) * (l2_interval(2)-l2_interval(1)) + l2_interval(1);
Rand_v = rand([50, 4]) * 4;
% Define particles
particles = [Rand_Sigma1, Rand_l1, Rand_Sigma2, Rand_l2, Rand_v, ones(50, 5)*3e8];
for iter = 1:iter_Max
    % Calculate MSE in current position
    fprintf("Iteraion %d starts\n", iter);
    MSE = zeros(1, 50);
    for par = 1: 50
        sigma1 = particles(par, 1);
        l1 = particles(par, 2);
        sigma2 = particles(par, 3);
        l2 = particles(par, 4);
        [K_all, ~] = kernel_PSO_SE([train_t; val_t], sigma1, l1, sigma2, l2);
        K = K_all(1:train_len, 1:train_len);
        K_s = K_all(train_len+1:train_len+validation_len, 1:train_len);
        y_bar = K_s*(K^-1)*train_y;
        MSE(par) = Error_cal(val_y, y_bar); %MSE cal
        if MSE(par) < particles(par, 13) %Update personal best
            particles(par, 9:13) = [sigma1, l1, sigma2, l2, MSE(par)];
        end
        if MSE(par) < Global_Best(5) %Update global best
            Global_Best = [sigma1, l1, sigma2, l2, MSE(par)];
        end
    end
    MSE_plot(iter) = Global_Best(5);
    fprintf("Current best %f \n", Global_Best(5));
    % Update velocity and position
    for par = 1:50
        v1 = particles(par, 9:12)-particles(par, 1:4);
        v2 = Global_Best(1:4)-particles(par, 1:4);
        new_v = w*particles(par, 5:8) + c1*rand()*v1 + c2*rand()*v2;
        new_v = min(new_v, v_max);
        new_v = max(new_v, -v_max);
        particles(par, 5:8) = new_v;
        particles(par, 1:4) = particles(par, 1:4) + new_v;
        particles(par, 1:4) = Boundary_Check(particles(par, 1:4), sigma1_interval, l1_interval, sigma2_interval, l2_interval);
    end
end
plot(MSE_plot)
%% Plot
tr_y = train_y;
sigma1 = Global_Best(1);
l1 = Global_Best(2);
sigma2 = Global_Best(3);
l2 = Global_Best(4);
len_tr = train_len;
len = total_len;
t = (1:len)';
[K_all, dis] = kernel_PSO_SE(t, sigma1, l1, sigma2, l2);
K = K_all(1:len_tr, 1:len_tr);
K_s = K_all(len_tr+1:len, 1:len_tr);
K_ss = K_all(len_tr+1:len, len_tr+1:len);
y_bar = K_s*(K^-1)*tr_y + data_ave;
var_y = K_ss - K_s/K*(K_s');
str = sprintf("%s, \\sigma_1=%f, l_1=%f, \\sigma_2=%f, l_2=%f",file_name, sigma1, l1, sigma2, l2);
figure;
subplot(1, 2, 1);
hold on;
plot(raw_data, 'b-x');
plot(len_tr+1:len_tr+validation_len, y_bar(1:validation_len), 'g-o');
plot(len_tr+validation_len+1:len, y_bar(validation_len+1:end), 'r-o');
plot(ones(1,length(raw_data))*data_ave, 'b');
legend("Original Data", "Validation", "Prediction", "Original Data Average");
xlabel("Trading Day");
ylabel("US Dollar($)");
title(str);
hold off;
subplot(1, 2, 2);
hold on;
plot(len_tr+1:len, raw_data(len_tr+1:len), 'b-x');
plot(len_tr+1:len_tr+validation_len, y_bar(1:validation_len), 'g-o');
plot(len_tr+validation_len+1:len, y_bar(validation_len+1:end), 'r-o');
plot(len_tr+1:len, ones(1,len-train_len)*data_ave, 'b');
MSE = Error_cal(y_bar(validation_len+1:end), raw_data(train_len+validation_len+1: end));
legend("Original Data", "Validation", "Prediction", "Original Data Average");
xlabel("Trading Day");
ylabel("US Dollar($)");
title(str+' Zoomed');
hold off;
disp(MSE);
%% Function
function out = Boundary_Check(pos, sig1, l1, sig2, l2)
    pos(1) = max(min(sig1(2), pos(1)), sig1(1));
    pos(2) = max(min(l1(2), pos(2)), l1(1));
    pos(3) = max(min(sig2(2), pos(3)), sig2(1));
    pos(4) = max(min(l2(2), pos(4)), l2(1));
    out = pos;
end