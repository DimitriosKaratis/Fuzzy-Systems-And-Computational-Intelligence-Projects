%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 2
% G_CarControl.pdf


% This function calculates vertical (dv) and horizontal (dh) distances
% from obstacles for a given position (x, y). The map is divided into 4
% regions based on x to assign distances differently.
function [dv, dh] = distance_sensors(x, y)

%% Region 1: x <= 10
if (x <= 10)
    dv = -y;  % Vertical distance: since y is negative, -y gives distance from top boundary
    
    % Horizontal distance depends on y
    if (y >= -5)
        dh = 10 - x;  
    elseif (y >= -6)
        dh = 11 - x;
    elseif (y >= -7)
        dh = 12 - x;
    else
        dh = 16 - x;  
    end

%% Region 2: 10 < x <= 11
elseif (x <= 11)
    dv = 5 - y;  
    
    if (y >= -6)
        dh  = 11 - x;
    elseif (y >= -7)
        dh = 12 - x;
    else
        dh = 16 - x;
    end

%% Region 3: 11 < x <= 12
elseif (x <= 12)
    dv = 6 - y;  
    
    if (y >= -7)
        dh = 12 - x;
    else
        dh = 16 - x;
    end

%% Region 4: 12 < x <= 15
elseif (x <= 15)
    dv = 7 - y;      % Vertical distance from boundary
    dh = 16 - x;     % Horizontal distance to map limit
end

%% Limit maximum distance to 1
if (dv > 1)
    dv = 1;  % Cap vertical distance
end

if (dh > 1)
    dh = 1;  % Cap horizontal distance
end

end
