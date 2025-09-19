%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 1
% 14_Satellite.pdf - Sxediasi Asafous Elegkti (FLC)

%% The fuzzy controller with INITIAL gains

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

% Convert the continuous-time system Gp(s) into a discrete-time plant Gp_d(z)
% using Zero-Order Hold (ZOH) method. 
% The ZOH assumes the control input is held constant between sampling instants,
% which matches how digital controllers typically work.
Gp_d = c2d(Gp, Ts, 'zoh');

%% Control Loop Simulation

N = 500;           % Number of simulation steps      
r_fuzzy = 60*ones(1, N); % Reference signal (step input of magnitude 60)

% Create discrete state-space model from discrete system Gp_d
[A, B, C, D] = ssdata(Gp_d);  

% Controller gains (tuning parameters)
% Increase Kd = reduces overshoot but increases rise time
% Increase Ke = increases overshoot but decreases rise time
Ke = 1.00;     % Gain for error input 
Kd = 0.667;    % Gain for delta_error input to fuzzy controller
K = 3.615;   % Output gain for fuzzy controller
e_scale = 60;  % Scaling factor for reference/input normalization 

% The fuzzy pi controller
show_info = true;
[y_fuzzy, u_fuzzy, ~, ~, ~] = fuzzyPI_simulation(N, Ts, r_fuzzy, fis, A, B, C, D, Ke, Kd, K, e_scale, show_info);

%% Plot results
time = (0:N-1)*Ts;

% Performance indicators
stepinfo(y_fuzzy, time, 60)


% Plot the fuzzy PI response
figure;
plot(time, y_fuzzy, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Output');
grid on;
hold on;

% Plot the linear PI
c = 1.5;
K = 2.41; 
Kp = K;          
Ki = K * c;     
% Transfer function of the closed loop system
Gc = K * tf([1 c], [1 0]);
sys_ol = Gc * Gp;
sys_cl = feedback(sys_ol, 1, -1);  

% Make the reference signal transpose
r_linear = 60 * ones(length(time), 1);  
% Simulate closed loop system output of the linear pi
y_linear = lsim(sys_cl, r_linear, time);


% Plot the linear PI response
plot(time, r_linear, '--', time, y_linear, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Output');
title('Step Responses of the PI Controllers');
legend('Fuzzy PI', 'Reference', 'Linear PI');
grid on;
hold off;



%% Plot control signal u(t)

% Plot the fuzzy PI signal u
figure;
plot(time, u_fuzzy, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u(t)');
grid on;
hold on;


% Plot the linear PI signal u
e = r_linear - y_linear;
u_linear = lsim(Gc, e, time);  % Control signal (u = Gc * e)
plot(time, u_linear, 'b', 'LineWidth', 1.5);

xlabel('Time (s)');
ylabel('u(t)');
title('Control signal u of the PI Controllers');
grid on;
legend('Fuzzy PI', 'Linear PI');
hold off;


