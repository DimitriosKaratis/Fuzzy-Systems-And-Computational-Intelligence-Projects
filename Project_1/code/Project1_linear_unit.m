%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 1
% 14_Satellite.pdf - Sxediasi Grammikou Elegkti

%% The Linear controller under unit step input

clc; clear; close all;

% System
Gp = tf(10, conv([1 1], [1 9]));  % Gp(s) = 10 / ((s+1)(s+9))

% PI controller parameters for the root locus
c = 1.5;  % Zero
K = 1;  % Inital value of the gain K

Gc = K * tf([1 c], [1 0]);  % PI controller: Gc(s) = K*(s+c)/s

% Transfer function of the open loop system
sys_ol = Gc * Gp;

% Root Locus
figure;
rlocus(sys_ol);
grid on;
sgrid(0.6, 1.6); % We need ζ > 0.59 and ωn > 1.5
title('Root Locus of the PI Controller');

% K value that I chose when looking at the root locus (we need ζ > 0.59 and
% ωn > 1.5, in order to satisfy the constrains)
K = 2.41; %1.49 isos an ta vro skoura me to allo

% PI parameter calculation
Kp = K;          
Ki = K * c;       

% Transfer function of the closed loop system
Gc = K * tf([1 c], [1 0]);
sys_ol = Gc * Gp;
sys_cl = feedback(sys_ol, 1, -1);  

% Create the step response figure
figure;
step(sys_cl, 0:0.01:5);
title('Step Response of the PI Controller');
grid on;

% Get relevant system information
stepinfo(sys_cl)


%% Simulate the controller signal (u) over time 

% Time sample
Ts = 0.01;
t = 0:Ts:5;

% Reference signal (unit step input)
r = 1 * ones(length(t), 1);  

% Simulate closed loop system output
y = lsim(sys_cl, r, t);

% Error
e = r - y;

% Control signal (u = Gc * e)
u = lsim(Gc, e, t);

% Control signal (u) plot
figure;
plot(t, u, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u(t)');
title('Control signal u of the linear PI controller');
grid on;
