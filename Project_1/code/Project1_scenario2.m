%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 1
% 14_Satellite.pdf - Sxediasi Asafous Elegkti (FLC)

%% SCENARIO 2

clc; clear; close all;

% Create Mamdani FIS
fis = createFIS_PI();

%% Discretize System in order to make the simulations
% Define the Laplace variable 's' for continuous-time transfer functions
s = tf('s');

% Define the continuous-time system: Gp(s) = 10 / ((s + 1)(s + 9))
Gp = 10 / ((s + 1)*(s + 9));

% Set the sampling time for the digital controller (10 ms per sample)
Ts = 0.01;  
T_end = 12;    
N = T_end / Ts;
time = (0:N-1) * Ts;

%% Step Reference A (Fig. 3)
% Define r(k) 
for k = 1:N
    t = (k-1)*Ts;
    if t < 4
        r(k) = 60;
    elseif t < 8
        r(k) = 20;
    else
        r(k) = 40;
    end
end

% Convert the continuous-time system Gp(s) into a discrete-time plant Gp_d(z)
% using Zero-Order Hold (ZOH) method. 
% The ZOH assumes the control input is held constant between sampling instants,
% which matches how digital controllers typically work.
Gp_d = c2d(Gp, Ts, 'zoh');

%% Control Loop Simulation
     
% Create discrete state-space model from discrete system Gp_d
[A, B, C, D] = ssdata(Gp_d);  

% Controller gains (tuning parameters)
% Increase Kd = reduces overshoot but increases rise time
% Increase Ke = increases overshoot but decreases rise time
Ke = 1.35;     % Gain for error input 
Kd = 0.334;    % Gain for delta_error input to fuzzy controller
K = 15.5445;   % Output gain for fuzzy controller
e_scale = 60;  % Scaling factor for reference/input normalization 

% The fuzzy pi controller
show_info = false;
[y, u, ~, ~, ~] = fuzzyPI_simulation(N, Ts, r, fis, A, B, C, D, Ke, Kd, K, e_scale, show_info);

% Plot 1 – FLC Response
figure;
plot(time, r, '--k', time, y, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('r(t), y(t)');
title('FLC Response A');
legend('r(t)', 'y(t)');
grid on;

% Plot 2 – Control Signal
figure;
plot(time, u, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u(t)');
title('Control Signal A');
grid on;

%% Ramp with plateau (Fig. 4)
T_end = 16;     
N = T_end / Ts; 
time = (0:N-1) * Ts;

% Create ramp reference
for k = 1:N
    t = (k-1) * Ts;
    if t < 5
        r(k) = 0.4 + (60 - 0.4)/5 * t;  % increasing ramp
    elseif t < 8
        r(k) = 60;                      % steady
    elseif t <= 16
        r(k) = 60 - (60 - 0.4)/(8) * (t - 8);  % decreasing ramp
    end
end

% The fuzzy pi controller
show_info = false;
[y, u, ~, ~, ~] = fuzzyPI_simulation(N, Ts, r, fis, A, B, C, D, Ke, Kd, K, e_scale, show_info);

% Plot 1: System response
figure;
plot(time, r, '--k', time, y, 'b', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('r(t), y(t)');
title('System Response B');
legend('r(t)', 'y(t)');
grid on;

% Plot 2: Control signal u(t)
figure;
plot(time, u, 'r', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('u(t)');
title('Control Signal B');
grid on;
