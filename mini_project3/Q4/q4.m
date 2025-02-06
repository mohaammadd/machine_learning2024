clc;
clear;
close all;

%% Initialization
M = 4; % Number of fuzzy rules (membership functions)
num_training = 200; % Number of training samples
total_num = 700; % Total number of samples
lambda = 0.1; % Learning rate for updating parameters

% Initialize matrices and vectors
x_bar = zeros(num_training, M);
g_bar = zeros(num_training, M);
sigma = zeros(num_training, M);
y = zeros(total_num, 1);
u = zeros(total_num, 1);
x = zeros(total_num, 1);
y_hat = zeros(total_num, 1);
f_hat = zeros(total_num, 1);
z = zeros(M, 1);
g_u = zeros(total_num, 1);

% Initial input and output
u(1) = -1 + 2 * rand; % Random initial input between -1 and 1
y(1) = 0;
y(2) = 0; % Avoid accessing y(-1) in the update equation
g_u(1) = 0.6 * sin(pi * u(1)) + 0.3 * sin(3 * pi * u(1)) + 0.1 * sin(5 * pi * u(1));
f_hat(1) = g_u(1);

%% Define initial membership functions
u_min = -1;
u_max = 1;
h = (u_max - u_min) / (M - 1);

for k = 1:M
    x_bar(1, k) = -1 + h * (k - 1);
    g_bar(1, k) = 0.6 * sin(pi * x_bar(1, k)) + 0.3 * sin(3 * pi * x_bar(1, k)) + 0.1 * sin(5 * pi * x_bar(1, k));
end

sigma(1, 1:M) = (max(x_bar(1, :)) - min(x_bar(1, :))) / M;

%% Training Process (Using Gradient Descent)
for q = 2:num_training
    b = 0; 
    a = 0;
    x(q) = -1 + 2 * rand; % Random input between -1 and 1
    u(q) = x(q);
    g_u(q) = 0.6 * sin(pi * u(q)) + 0.3 * sin(3 * pi * u(q)) + 0.1 * sin(5 * pi * u(q));

    % Compute fuzzy system output
    for l = 1:M
        z(l) = exp(-((x(q) - x_bar(q-1, l)) / sigma(q-1, l))^2);
        b = b + z(l);
        a = a + g_bar(q-1, l) * z(l);
    end

    f_hat(q) = a / b;
    y(q+1) = 0.3 * y(q) + 0.6 * y(q-1) + g_u(q);
    y_hat(q+1) = 0.3 * y(q) + 0.6 * y(q-1) + f_hat(q);

    % Update fuzzy rule parameters using gradient descent
    for l = 1:M
        g_bar(q, l) = g_bar(q-1, l) - lambda * (f_hat(q) - g_u(q)) * z(l) / b;
        x_bar(q, l) = x_bar(q-1, l) - lambda * ((f_hat(q) - g_u(q)) / b) * ...
            (g_bar(q-1, l) - f_hat(q)) * z(l) * 2 * (x(q) - x_bar(q-1, l)) / (sigma(q-1, l)^2);
        sigma(q, l) = sigma(q-1, l) - lambda * ((f_hat(q) - g_u(q)) / b) * ...
            (g_bar(q-1, l) - f_hat(q)) * z(l) * 2 * (x(q) - x_bar(q-1, l)) / (sigma(q-1, l)^3);
    end
end

%% Testing the Model with New Data
for q = num_training:total_num-1
    b = 0; 
    a = 0;
    x(q) = sin(2 * q * pi / 200); % Generate test input data
    u(q) = x(q);
    g_u(q) = 0.6 * sin(pi * u(q)) + 0.3 * sin(3 * pi * u(q)) + 0.1 * sin(5 * pi * u(q));

    % Compute fuzzy system output for test data
    for l = 1:M
        z(l) = exp(-((x(q) - x_bar(num_training-1, l)) / sigma(num_training-1, l))^2);
        b = b + z(l);
        a = a + g_bar(num_training-1, l) * z(l);
    end

    f_hat(q) = a / b;
    y(q+1) = 0.3 * y(q) + 0.6 * y(q-1) + g_u(q);
    y_hat(q+1) = 0.3 * y(q) + 0.6 * y(q-1) + f_hat(q);
end

%% Plot results
figure;
plot(1:total_num, y, 'b', 1:total_num, y_hat, 'r:', 'LineWidth', 2);
legend('System Output', 'Fuzzy Model Output')
xlabel('Time Steps');
ylabel('Output');
title('System vs. Fuzzy Model Output');
grid on;

figure;
xp = -2:0.001:2;
for l = 1:M
    miu_x = exp(-((xp - x_bar(1, l)) ./ sigma(1, l)).^2);
    plot(xp, miu_x, 'LineWidth', 2);
    hold on;
end
xlabel('u');
ylabel('Initial Membership Functions');
title('Initial Membership Functions');
grid on;

figure;
for l = 1:M
    miu_x = exp(-((xp - x_bar(num_training-1, l)) ./ sigma(num_training-1, l)).^2);
    plot(xp, miu_x, 'LineWidth', 2);
    hold on;
end
xlabel('u');
ylabel('Final Membership Functions');
title('Final Membership Functions After Training');
grid on;

