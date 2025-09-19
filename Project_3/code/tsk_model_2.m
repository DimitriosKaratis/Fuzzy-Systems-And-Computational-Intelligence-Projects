%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 3
% Regression.pdf
%% Part 1: Airfoil Self-Noise Dataset (UCI) 
% TSK Model 2 -> 3 MFs, Singleton output


clear all; close all; clc;   

% Start time
tic

%% Load dataset
load airfoil_self_noise.dat         % Load airfoil dataset
data = airfoil_self_noise;          % Assign to 'data' (5 inputs + 1 output)
data = unique(data,'rows');         % Remove duplicate rows

% Store original output min/max for later denormalization
y_min = min(data(:,6));
y_max = max(data(:,6));

%% Normalize inputs & output to [0,1]
for i = 1:6
    data_min = min(data(:,i));
    data_max = max(data(:,i));
    data(:,i) = (data(:,i) - data_min) / (data_max - data_min);  % Min-max normalization
end

%% Experiment parameters
numRuns = 30;                         % Number of runs for multi-run experiment
metrics = zeros(numRuns,5);           % Store metrics for each run [MSE RMSE NMSE NDEI R2]
allSeeds = zeros(numRuns,1);          % Store random seeds for reproducibility

fprintf('TSK Model 2 -> 3 MFs, Singleton output\n');

%% Loop over multiple runs
for run = 1:numRuns
    fprintf('--- Run %d/%d ---\n', run, numRuns);

    % Set random seed for reproducibility
    seed = randi(1e6);
    rng(seed);
    allSeeds(run) = seed;

    % Shuffle and split data into training, validation, and test sets
    N = length(data);
    idx = randperm(N);
    data_shuffled = data(idx,:);

    trainingData   = data_shuffled(1 : round(N*0.6) , :);       % 60% training
    validationData = data_shuffled(round(N*0.6) + 1 : round(N*0.8) , :); % 20% validation
    checkData      = data_shuffled(round(N*0.8) + 1 : end , :); % 20% testing

    % Generate initial FIS using grid partitioning (3 MFs per input)
    opt = genfisOptions('GridPartition');
    opt.NumMembershipFunctions = [3 3 3 3 3];                   % 3 MFs per input
    opt.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf" "gbellmf"];
    opt.OutputMembershipFunctionType = 'constant';              

    fis = genfis(trainingData(:,1:5), trainingData(:,6), opt);  % Generate initial FIS

    % ANFIS training options
    opt = anfisOptions;
    opt.InitialFIS = fis;
    opt.EpochNumber = 200;                                      
    opt.ValidationData = validationData;
    opt.DisplayANFISInformation = 0;                               
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;

    % Train ANFIS
    [~, ~ ,~ ,chkFIS ,~] = anfis(trainingData,opt);

    % Evaluate ANFIS on test data
    output = evalfis(chkFIS, checkData(:,1:5));

    % Denormalize predictions & true outputs
    output_raw = output * (y_max - y_min) + y_min;
    check_raw  = checkData(:,6) * (y_max - y_min) + y_min;

    % Compute prediction errors and performance metrics
    error_raw = check_raw - output_raw;
    MSE = mean(error_raw.^2);
    RMSE = sqrt(MSE);

    SSres = sum((check_raw - output_raw).^2);              % Residual sum of squares
    SStot = sum((check_raw - mean(check_raw)).^2);         % Total sum of squares
    R2 = 1 - SSres/SStot;                                  % Coefficient of determination

    NMSE = 1 - R2;                                         % Normalized MSE
    NDEI = sqrt(NMSE);                                     % Normalized root MSE (NDEI)

    % Store metrics in matrix
    metrics(run,:) = [MSE, RMSE, NMSE, NDEI, R2];

    % Print metrics for this run
    fprintf('MSE = %f  RMSE = %f  NMSE = %f  NDEI = %f  R2 = %f\n\n', ...
            MSE, RMSE, NMSE, NDEI, R2);
end

%% Compute statistics across runs
mean_metrics = mean(metrics,1);         % Mean of metrics
std_metrics  = std(metrics,0,1);        % Standard deviation of metrics

fprintf('\n==== Summary over %d runs ====\n', numRuns);
fprintf('MSE:  mean = %.4f, std = %.4f\n', mean_metrics(1), std_metrics(1));
fprintf('RMSE: mean = %.4f, std = %.4f\n', mean_metrics(2), std_metrics(2));
fprintf('NMSE: mean = %.4f, std = %.4f\n', mean_metrics(3), std_metrics(3));
fprintf('NDEI: mean = %.4f, std = %.4f\n', mean_metrics(4), std_metrics(4));
fprintf('R2:   mean = %.4f, std = %.4f\n', mean_metrics(5), std_metrics(5));

%% Find representative run (closest to mean R2)
[~, best_idx] = min(abs(metrics(:,5) - mean_metrics(5)));
best_seed = allSeeds(best_idx);

fprintf(['\nRepresentative run: %d (seed = %d):\n' ...
         'MSE = %.3f\n' ...
         'RMSE = %.3f\n' ...
         'NMSE = %.3f\n' ...
         'NDEI = %.3f\n' ...
         'R2 = %.3f\n'], ...
        best_idx, best_seed, metrics(best_idx,1), metrics(best_idx,2), ...
        metrics(best_idx,3), metrics(best_idx,4), metrics(best_idx,5));

%% Shuffle & split again for representative run plotting
N = length(data);
idx = randperm(N);
data_shuffled = data(idx,:);

trainingData   = data_shuffled(1 : round(N*0.6) , :);
validationData = data_shuffled(round(N*0.6) + 1 : round(N*0.8) , :);
checkData      = data_shuffled(round(N*0.8) + 1 : end , :);

% Generate FIS again with 3 MFs per input
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [3 3 3 3 3];
opt.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf" "gbellmf"];
opt.OutputMembershipFunctionType = 'constant';
fis = genfis(trainingData(:,1:5), trainingData(:,6), opt);

% --- Plot initial membership functions (MFs) ---
figure;
for i = 1:5
    [x,mf] = plotmf(fis,'input',i);     
    subplot(3,2,i)
    plot(x,mf)
    xlabel(['Input ' num2str(i)])
end
sgtitle('TSK model 2: MFs before Training')
saveas(gcf, 'TSK_model_2_imgs/2_MFs_before_Training.png')

% ANFIS training options
opt = anfisOptions;
opt.InitialFIS = fis;
opt.EpochNumber = 200;
opt.ValidationData = validationData;
opt.DisplayANFISInformation = 0;
opt.DisplayErrorValues = 0;
opt.DisplayStepSize = 0;
opt.DisplayFinalResults = 0;

% Train ANFIS
[trnFIS,trainError,stepSize,chkFIS,chkError] = anfis(trainingData,opt);

% --- Plot final MFs after training ---
figure;
for i = 1:5
    [x,mf] = plotmf(chkFIS,'input',i);
    subplot(3,2,i)
    plot(x,mf)
    xlabel(['Input ' num2str(i)])
end
sgtitle('TSK model 2: MFs after Training')
saveas(gcf, 'TSK_model_2_imgs/2_MFs_after_Training.png')

% Evaluate representative run
output = evalfis(chkFIS, checkData(:,1:5));
output_raw = output * (y_max - y_min) + y_min;
check_raw  = checkData(:,6) * (y_max - y_min) + y_min;
error_raw  = check_raw - output_raw;

% --- Plot learning curve ---
[minValError, bestEpoch] = min(chkError);   % Find epoch with minimum validation error

figure;
plot(1:length(trainError), trainError, 'b', 'LineWidth', 1.5); hold on;
plot(1:length(chkError), chkError, 'r', 'LineWidth', 1.5);
plot(bestEpoch, minValError, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'g'); 
xline(bestEpoch, '--k', sprintf('Best Epoch = %d', bestEpoch));
xlabel('Epoch'); ylabel('Error');
legend('Training Error', 'Validation Error', 'Best chkFIS');
title('Representative Run (Model 2): Learning Curve');
grid on;
saveas(gcf, 'TSK_model_2_imgs/2_Learning_Curve.png');

% --- Plot prediction errors for representative run ---
figure;
plot(error_raw)
title('TSK model 2: Prediction Errors (Raw Data)')
xlabel('Sample Index')   
ylabel('Error')          
saveas(gcf,'TSK_model_2_imgs/2_Pred_Errors.png')

% Stop time
toc
