%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 1
% 14_Satellite.pdf - Sxediasi Asafous Elegkti (FLC)

%% fuzzyPI_simulation φunction σimulates a discrete-time fuzzy PI controller

function [y, u, du, e, de] = fuzzyPI_simulation(N, Ts, r, fis, A, B, C, D, Ke, Kd, K, e_scale, show_info)
%
% Inputs:
%   x   - initial state vector (zeros(size(e_scaled,1),1))
%   N   - number of simulation steps
%   Ts  - sampling time
%   r   - reference signal (1xN)
%   fis - fuzzy inference system (Mamdani FIS)
%   A, B, C, D - discrete state-space matrices
%   Ke, Kd, K - controller gains
%   e_scaled   - scaling factor
%
% Outputs:
%   y   - system output
%   u   - control signal
%   du  - incremental control output (from fuzzy controller)
%   e   - tracking error
%   de  - change in error

% Initialize signals
y = zeros(1,N);
u = zeros(1,N);
du = zeros(1,N);
e = zeros(1,N);
de = zeros(1,N);
x = zeros(size(A,1), 1);  % Initialize state vector x for the system, 
                          % Number of states = number of rows in A

for k = 2:N
    % Compute error and change in error
    e(k) = r(k) - y(k-1);
    e(k) = e(k) / e_scale;
    
    de(k) = e(k) - e(k-1);
    
    % Scale inputs
    e_scaled = (Ke * e(k));
    de_scaled = (Kd * de(k) / Ts);

    % Clamp values to [-1, 1]
    e_clamped = max(min(e_scaled, 1), -1);
    de_clamped = max(min(de_scaled, 1), -1);
    
    % Fuzzy controller output
    du_fuzzy = evalfis(fis, [e_clamped, de_clamped]);
    
    % Compute incremental control
    du(k) = K * du_fuzzy * Ts * e_scale;
    
    % Accumulate control signal
    u(k) = u(k-1) + du(k);
    
    % Update system state and output
    x = A*x + B*u(k);
    y(k) = C*x + D*u(k);

    % Print information if needed
    if mod(k, 1) == 0 && show_info == true
        fprintf("Step %d → e=%.3f, de=%.3f → du=%.3f, u=%.3f, y=%.3f, de_sc=%.3f, de_cl=%.3f\n", ...
        k, e(k), de(k), du(k), u(k), y(k), de_scaled, de_clamped);
    end
end

end
