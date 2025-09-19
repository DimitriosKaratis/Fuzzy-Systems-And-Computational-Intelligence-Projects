%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 2
% G_CarControl.pdf
%% Car Controller for the FINAL Fuzzy Inference System (FIS)

clear; clc; close all;  

%% Initializing variables
x_init = 9.1;          % Initial x-coordinate of the car
y_init = -4.3;         % Initial y-coordinate of the car
u = 0.05;              % Car's velocity
thetas = [0 45 90];    % Initial orientations (in degrees)
x_desired = 15;        % Target x-coordinate
y_desired = -7.2;      % Target y-coordinate
threshold = 0.05;      % Distance threshold to consider the car has reached the target

%% Read FIS 
% Change to 'myfis.fis' to see the output of the original version of the fis
carFIS = readfis('myfis_optimized.fis');  
fprintf('FOR THE OPTIMIZED FIS:\n\n'); 

%% Route Simulation for each initial orientation
for i = 1 : 1 : length(thetas)
    
    % Reset car position and orientation for this simulation
    x = x_init;
    y = y_init;
    theta = thetas(i);
    
    % Initialize arrays to store the car's trajectory
    x_moves = []; 
    y_moves = [];
    
    flag = 1;       % Flag to check if the car is still inside map boundaries
    isClose = 0;    % Flag to check if the car reached the target
    
    % Run simulation until car leaves map or reaches target
    while (flag == 1 && isClose == 0)
       
        % Calculate distances from obstacles 
        [dv, dh] = distance_sensors(x, y);
        
        % Use FLC to estimate steering output 
        delta_theta = evalfis(carFIS, [dv dh theta]);
        
        % Update car heading
        theta = theta + delta_theta;
        
        % Move the car forward in the direction of the new heading
        x = x + u * cosd(theta);
        y = y + u * sind(theta);
        
        % Check if car is still within map boundaries
        if (x < 0) || (x > 15) || (y > 0) || (y < -8)
            flag = 0;  % Stop simulation if car leaves map
        end
        
        % Store the updated position for plotting
        x_moves = [x_moves; x];
        y_moves = [y_moves; y];
        
        % Check if car reached the desired target within the threshold
        if (sqrt((abs(x - x_desired))^2 + (abs(y - y_desired))^2) < threshold)
            isClose = 1;  % Stop simulation if target reached
        end
        
    end
    
    % Print the final position of the car
    fprintf('Theta %d° -> Final Position: (%.6f, %.6f) | Desired: (%.2f, %.2f)\n', ...
        thetas(i), x, y, x_desired, y_desired);

    %% Create map and plot results
    obstacle_x = [10; 10; 11; 11; 12; 12; 15];  % X-coordinates of map obstacles
    obstacle_y = [0; -5; -5; -6; -6; -7; -7];   % Y-coordinates of map obstacles
    
    figure; 
    line(x_moves, y_moves, 'Color', 'blue');         % Plot car trajectory
    line(obstacle_x, obstacle_y, 'Color', 'black');  % Plot map obstacles
    
    hold on;
    plot(x_init, y_init, 'O');        % Mark starting position
    plot(x_desired, y_desired, 'X');  % Mark desired target position
      
    title(['Degrees: ', num2str(thetas(i))]);  
    
    % Calculate final errors 
    error_x = x_desired - x;
    error_y = y_desired - y;

    % Calculate percentage errors relative to desired values
    error_x_percent = (error_x / x_desired) * 100;
    error_y_percent = (error_y / y_desired) * 100;
    
    % Print error percentages
    fprintf('Error: x = %.5f%%, y = %.5f%%\n\n', error_x_percent, error_y_percent);
end
