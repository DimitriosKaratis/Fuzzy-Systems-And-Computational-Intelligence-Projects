%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 1
% 14_Satellite.pdf - Sxediasi Asafous Elegkti (FLC)

%% Function to create the FIS

function fis = createFIS_PI()   
    
    fis = mamfis( ...
        'Name', 'FLC_PI', ...
        'AndMethod', 'min', ...
        'OrMethod', 'max', ...
        'ImplicationMethod', 'prod', ...    
        'AggregationMethod', 'max', ...
        'DefuzzificationMethod', 'centroid' ... 
    );
    
    A_scaled = 1.0;
    %% Input 1: error [-1, 1]
    fis = addInput(fis, [-A_scaled A_scaled], 'Name', 'error');
    fis = addMF(fis, 'error', 'trimf', [-A_scaled    -A_scaled      -3*A_scaled/6], 'Name', 'NL');
    fis = addMF(fis, 'error', 'trimf', [-A_scaled    -4*A_scaled/6  -A_scaled/6],  'Name', 'NM');
    fis = addMF(fis, 'error', 'trimf', [-5*A_scaled/6 -2*A_scaled/6    A_scaled/6],    'Name', 'NS');
    fis = addMF(fis, 'error', 'trimf', [-3*A_scaled/6   0       3*A_scaled/6],  'Name', 'ZR');
    fis = addMF(fis, 'error', 'trimf', [-A_scaled/6      2*A_scaled/6     5*A_scaled/6],'Name', 'PS');
    fis = addMF(fis, 'error', 'trimf', [A_scaled/6    4*A_scaled/6   A_scaled],    'Name', 'PM');
    fis = addMF(fis, 'error', 'trimf', [3*A_scaled/6  A_scaled     A_scaled],    'Name', 'PL');
    
    %% Input 2: delta_error [-1, 1]
    fis = addInput(fis, [-A_scaled A_scaled], 'Name', 'delta_error');
    fis = addMF(fis, 'delta_error', 'trimf', [-A_scaled    -A_scaled        -5*A_scaled/8], 'Name', 'NV');
    fis = addMF(fis, 'delta_error', 'trimf', [-A_scaled    -6*A_scaled/8    -3*A_scaled/8], 'Name', 'NL');
    fis = addMF(fis, 'delta_error', 'trimf', [-7*A_scaled/8 -4*A_scaled/8   -A_scaled/8], 'Name', 'NM');
    fis = addMF(fis, 'delta_error', 'trimf', [-5*A_scaled/8 -2*A_scaled/8    A_scaled/8],     'Name', 'NS');
    fis = addMF(fis, 'delta_error', 'trimf', [-3*A_scaled/8  0        3*A_scaled/8], 'Name', 'ZR');
    fis = addMF(fis, 'delta_error', 'trimf', [-A_scaled/8       2*A_scaled/8    5*A_scaled/8], 'Name', 'PS');
    fis = addMF(fis, 'delta_error', 'trimf', [A_scaled/8   4*A_scaled/8    7*A_scaled/8], 'Name', 'PM');
    fis = addMF(fis, 'delta_error', 'trimf', [3*A_scaled/8   6*A_scaled/8    A_scaled],     'Name', 'PL');
    fis = addMF(fis, 'delta_error', 'trimf', [5*A_scaled/8   A_scaled   A_scaled],     'Name', 'PV');
    
    %% Output: delta_u [-1, 1]
    fis = addOutput(fis, [-A_scaled A_scaled], 'Name', 'delta_u');
    fis = addMF(fis, 'delta_u', 'trimf', [-A_scaled    -A_scaled        -5*A_scaled/8], 'Name', 'NV');
    fis = addMF(fis, 'delta_u', 'trimf', [-A_scaled    -6*A_scaled/8    -3*A_scaled/8], 'Name', 'NL');
    fis = addMF(fis, 'delta_u', 'trimf', [-7*A_scaled/8 -4*A_scaled/8   -A_scaled/8], 'Name', 'NM');
    fis = addMF(fis, 'delta_u', 'trimf', [-5*A_scaled/8 -2*A_scaled/8    A_scaled/8],     'Name', 'NS');
    fis = addMF(fis, 'delta_u', 'trimf', [-3*A_scaled/8  0        3*A_scaled/8], 'Name', 'ZR');
    fis = addMF(fis, 'delta_u', 'trimf', [-A_scaled/8       2*A_scaled/8    5*A_scaled/8], 'Name', 'PS');
    fis = addMF(fis, 'delta_u', 'trimf', [A_scaled/8   4*A_scaled/8    7*A_scaled/8], 'Name', 'PM');
    fis = addMF(fis, 'delta_u', 'trimf', [3*A_scaled/8   6*A_scaled/8    A_scaled],     'Name', 'PL');
    fis = addMF(fis, 'delta_u', 'trimf', [5*A_scaled/8   A_scaled   A_scaled],     'Name', 'PV');
    
    %% Rules (7 x 9 = 63)
    
        rule_names = [
        % en = 1  % en = 2  % en = 3  % en = 4  % en = 5  % en = 6  % en = 7
        %  (NL)   %  (NM)   %  (NS)   %  (ZR)   %  (PS)   %  (PM)   %  (PL)
          1         1         1         1         2         3         4;    % Δen = 1 (NV)
          1         1         1         2         3         4         5;    % Δen = 2 (NL)
          1         1         2         3         4         5         6;    % Δen = 3 (NM)
          1         2         3         4         5         6         7;    % Δen = 4 (NS)
          2         3         4         5         6         7         8;    % Δen = 5 (ZR)
          3         4         5         6         7         8         9;    % Δen = 6 (PS)
          4         5         6         7         8         9         9;    % Δen = 7 (PM)
          5         6         7         8         9         9         9;    % Δen = 8 (PL)
          6         7         8         9         9         9         9];   % Δen = 9 (PV)
    
    
    ruleList = [];
    for i = 1:9      % Δen (rows, 9 MF)
        for j = 1:7  % en (cols, 7 MF)
            outIdx = rule_names(i,j);
            % Rule format: [en Δen output weight ANDmethod]
            ruleList = [ruleList; j i outIdx 1 1];
        end
    end
    
    fis = addRule(fis, ruleList);

    % Plot FIS control surface
    figure;
    gensurf(fis);
    title('Control Surface FLC (7x9x9)');
    
    % Save fis for later use
    writeFIS(fis, 'Ergasia1_fis.fis');
end