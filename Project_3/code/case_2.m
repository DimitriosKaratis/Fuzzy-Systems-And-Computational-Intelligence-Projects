%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 3
% Regression.pdf (Superconductivity dataset (UCI)) 
%% Part 2: Superconductivity dataset (UCI)

clear all; close all; clc;

%% Load Dataset 
% First 81 columns = features, last column = target (critical temperature)
load superconduct.csv; 
data = superconduct;
data = unique(superconduct, 'rows');  % remove duplicate rows

X = data(:,1:end-1);  % features
y = data(:,end);      % target

%% Store min/max for denormalization 
y_min = min(y);
y_max = max(y);

%% Normalize features and target to [0,1] 
X = normalize(X, "range");
y = (y - y_min) / (y_max - y_min);

%% Split Dataset: Train / Validation / Test (60/20/20) 
N = length(y);
rng(42);  % fixed seed for reproducibility
idx = randperm(N);  % shuffle indices
X = X(idx,:); y = y(idx);

Xtrain = X(1:round(0.6*N),:);  ytrain = y(1:round(0.6*N));
Xval   = X(round(0.6*N)+1:round(0.8*N),:); yval = y(round(0.6*N)+1:round(0.8*N));
Xtest  = X(round(0.8*N)+1:end,:); ytest = y(round(0.8*N)+1:end);

%% Parameters for Grid Search 
feature_set = [4 8 12 16 20 30];       % number of top features to evaluate
ra_set      = [0.2 0.3 0.4 0.5 0.6 0.7 0.8]; % cluster influence range values for subtractive clustering
numFolds    = 5;         % 5-fold cross-validation

%% Feature Selection using ReliefF 
% ReliefF ranks features based on importance relative to target
[ranked_features,~] = relieff(Xtrain, ytrain, 100);  % k=100 nearest neighbors for large dataset

%% Grid Search with 5-fold Cross-Validation 
results = [];
for nf = feature_set
    % Select top-nf features
    feat_idx = ranked_features(1:nf);
    Xtr_sel = Xtrain(:,feat_idx);

    for ra = ra_set
        fprintf("Testing nf=%d, ra=%.2f\n", nf, ra);

        cvMSE = zeros(numFolds,1);  % store MSE for each fold
        cv = cvpartition(length(ytrain), 'KFold', numFolds);

        for fold = 1:numFolds
            tr_idx = training(cv,fold);
            val_idx = test(cv,fold);

            Xtr_cv = Xtr_sel(tr_idx,:); ytr_cv = ytrain(tr_idx);
            Xval_cv = Xtr_sel(val_idx,:); yval_cv = ytrain(val_idx);

            % Subtractive Clustering Options 
            sc_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',ra);

            % Generate Initial FIS using Subtractive Clustering 
            fis = genfis(Xtr_cv, ytr_cv, sc_opt);

            % Count number of fuzzy rules generated
            numRules = length(fis.Rules);

            % ANFIS Training for CV
            % 100 epochs
            anfis_opt = anfisOptions('InitialFIS',fis, ...
                                     'EpochNumber',100, ...
                                     'ValidationData',[Xval_cv yval_cv], ...
                                     'DisplayErrorValues',0, ...
                                     'DisplayFinalResults',0);
            [~,~,~,~,chkError] = anfis([Xtr_cv ytr_cv], anfis_opt);

            % Store best CV error for this fold
            cvMSE(fold) = min(chkError);
        end

        % Compute mean CV MSE across folds
        meanMSE = mean(cvMSE);
        results = [results; nf ra numRules meanMSE];  % store results
        fprintf("Mean CV MSE = %.4f, NumRules = %d\n", meanMSE, numRules);
    end
end

%% Save Grid Search Results
results_table = array2table(results, ...
    'VariableNames',{'NumFeatures','ClusterRadius','NumRules','MeanCV_MSE'});
writetable(results_table,'TSK_superconduct_results/GridSearch_Results.csv');

%% Select Best Parameters 
% Choose configuration with minimum mean cv mse
[~,best_idx] = min(results(:,4));
best_nf = results(best_idx,1);
best_ra = results(best_idx,2);
fprintf("\nBest parameters: nf=%d, ra=%.2f\n", best_nf, best_ra);

%% Save selected features 
selected_features = ranked_features(1:best_nf);
writematrix(selected_features, 'TSK_superconduct_results/Selected_Features.csv');

%% Retrain Final ANFIS Model on Train + Validation 
feat_idx = ranked_features(1:best_nf);
Xtr_final = [Xtrain(:,feat_idx); Xval(:,feat_idx)];
ytr_final = [ytrain; yval];

% Generate initial FIS for final training
sc_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',best_ra);
fis = genfis(Xtr_final, ytr_final, sc_opt);

%% Plot Initial Membership Functions (Before ANFIS Updates) 
numInputs = length(fis.Inputs);
numToPlot = min(numInputs, 6);  % only plot up to 6 inputs

figure;
for i = 1:numToPlot
    subplot(ceil(numToPlot/2), 2, i);
    hold on;
    for j = 1:length(fis.Inputs(i).MembershipFunctions)
        mf = fis.Inputs(i).MembershipFunctions(j);
        x_range = linspace(0,1,100);
        y_mf = evalmf(mf, x_range);  
        plot(x_range, y_mf, 'LineWidth', 1.5);
    end
    title(sprintf('Input %d Initial MFs', i));
    xlabel('Normalized Feature Value'); ylabel('Membership Degree');
    legend({fis.Inputs(i).MembershipFunctions.Name}, 'Interpreter','none');
    grid on;
end
sgtitle('Initial Membership Functions of FIS Inputs (max 6)');
saveas(gcf,'TSK_superconduct_results/InitialFIS_MFs.png');

%% Train Final ANFIS Model 
% Use more epochs (200) for final model to ensure convergence
anfis_opt = anfisOptions('InitialFIS',fis, ...
                         'EpochNumber',200, ...
                         'ValidationData',[Xval(:,feat_idx) yval], ...
                         'DisplayErrorValues',0, ...
                         'DisplayFinalResults',0);

[trnFIS,trainError,stepSize,chkFIS,chkError] = anfis([Xtr_final ytr_final], anfis_opt);

%% Evaluate on Test Set
y_pred_norm = evalfis(chkFIS, Xtest(:,feat_idx));

% Denormalize predictions and true values
y_pred = y_pred_norm * (y_max - y_min) + y_min;
y_true = ytest * (y_max - y_min) + y_min;

%% Compute Performance Metrics
error_raw = y_true - y_pred;
MSE  = mean(error_raw.^2);
RMSE = sqrt(MSE);
SSres = sum((y_true - y_pred).^2);
SStot = sum((y_true - mean(y_true)).^2);
R2   = 1 - SSres/SStot;
NMSE = 1 - R2;
NDEI = sqrt(NMSE);

fprintf("\n--- Test Results (Denormalized) ---\n");
fprintf("MSE  = %.4f\n", MSE);
fprintf("RMSE = %.4f\n", RMSE);
fprintf("R2   = %.4f\n", R2);
fprintf("NMSE = %.4f\n", NMSE);
fprintf("NDEI = %.4f\n", NDEI);
fprintf("Best parameters: nf=%d, ra=%.2f\n", best_nf, best_ra);

% Save metrics
metrics_table = table(MSE,RMSE,R2,NMSE,NDEI);
writetable(metrics_table,'TSK_superconduct_results/FinalModel_Metrics.csv');

%% === Plot True vs Predicted ===
figure;
plot(y_true,'b'); hold on;
plot(y_pred,'r');
legend('True','Predicted');
title('True vs Predicted (Test Set, Denormalized)');
xlabel('Sample'); ylabel('Critical Temperature');
saveas(gcf,'TSK_superconduct_results/True_vs_Predicted.png');

%% Plot Prediction Errors 
figure;
plot(error_raw);
title('Prediction Errors (Denormalized)');
xlabel('Sample'); ylabel('Error');
saveas(gcf,'TSK_superconduct_results/Prediction_Errors.png');

%% Plot Learning Curve
[minValError, bestEpoch] = min(chkError);
figure;
plot(1:length(trainError), trainError, 'b', 'LineWidth',1.5); hold on;
plot(1:length(chkError), chkError, 'r', 'LineWidth',1.5);
plot(bestEpoch, minValError, 'ko','MarkerSize',8,'MarkerFaceColor','g');
xline(bestEpoch, '--k', sprintf('Best Epoch = %d', bestEpoch));
xlabel('Epoch'); ylabel('Error');
legend('Training Error','Validation Error','Best chkFIS');
title('Learning Curve (Best Model)');
grid on;
saveas(gcf,'TSK_superconduct_results/Learning_Curve.png');

%% Plot Final Membership Functions (After ANFIS Training) 
numInputs = length(chkFIS.Inputs);
numToPlot = min(numInputs, 6);  % only plot up to 6 inputs

figure;
for i = 1:numToPlot
    subplot(ceil(numToPlot/2), 2, i);
    hold on;
    for j = 1:length(chkFIS.Inputs(i).MembershipFunctions)
        mf = chkFIS.Inputs(i).MembershipFunctions(j);
        x_range = linspace(0,1,100);
        y_mf = evalmf(mf, x_range);  
        plot(x_range, y_mf, 'LineWidth', 1.5);
    end
    title(sprintf('Input %d MFs', i));
    xlabel('Normalized Feature Value'); ylabel('Membership Degree');
    legend({chkFIS.Inputs(i).MembershipFunctions.Name}, 'Interpreter','none');
    grid on;
end
sgtitle('Membership Functions of Final FIS Inputs (max 6)');
saveas(gcf,'TSK_superconduct_results/FinalFIS_MFs.png');

%% 3D Plot of Grid Search Results
figure;
numFeatures = results(:,1);
numRules    = results(:,3);
meanCV_MSE  = results(:,4);

scatter3(numFeatures, numRules, meanCV_MSE, 80, meanCV_MSE, 'filled'); 
colormap jet; colorbar;
xlabel('Number of Features'); ylabel('Number of Rules'); zlabel('Mean CV MSE');
title('Grid Search Results: Features vs Rules vs Mean CV MSE');
grid on;
view(45,30);
saveas(gcf,'TSK_superconduct_results/GridSearch_3DPlot.png');


%% CV MSE vs Number of Rules (for fixed NumFeatures)

% Load results
results_table = readtable('TSK_superconduct_results/GridSearch_Results.csv');

feature_list = unique(results_table.NumFeatures);
nFeat = length(feature_list);

figure;
for i = 1:nFeat
    nf = feature_list(i);

    % Select rows for this number of features
    idx = results_table.NumFeatures == nf;
    numRules = results_table.NumRules(idx);
    cvMSE = results_table.MeanCV_MSE(idx);

    % Sort by number of rules
    [numRules_sorted, sortIdx] = sort(numRules);
    cvMSE_sorted = cvMSE(sortIdx);

    % Subplot
    subplot(ceil(sqrt(nFeat)), ceil(sqrt(nFeat)), i);
    plot(numRules_sorted, cvMSE_sorted, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Rules');
    ylabel('Mean CV MSE');
    title(sprintf('Features = %d', nf));
    grid on;
end
sgtitle('CV MSE vs Rules for fixed Features');
saveas(gcf, 'TSK_superconduct_results/CVMSE_vs_Rules_allFeatures.png');


%% CV MSE vs Number of Features (for fixed NumRules)
rules_list = unique(results_table.NumRules);
nRules = length(rules_list);

% Pre-count how many valid plots we have (with >= 2 points)
valid_rules = [];
for i = 1:nRules
    nr = rules_list(i);
    idx = results_table.NumRules == nr;
    numFeatures = results_table.NumFeatures(idx);
    if numel(unique(numFeatures)) >= 2
        valid_rules(end+1) = nr; 
    end
end

nValid = length(valid_rules);

figure;
for i = 1:nValid
    nr = valid_rules(i);

    % Select rows for this number of rules
    idx = results_table.NumRules == nr;
    numFeatures = results_table.NumFeatures(idx);
    cvMSE = results_table.MeanCV_MSE(idx);

    % Sort by number of features
    [numFeatures_sorted, sortIdx] = sort(numFeatures);
    cvMSE_sorted = cvMSE(sortIdx);

    % Subplot
    subplot(ceil(sqrt(nValid)), ceil(sqrt(nValid)), i);
    plot(numFeatures_sorted, cvMSE_sorted, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Features');
    ylabel('Mean CV MSE');
    title(sprintf('Rules = %d', nr));
    grid on;
end
sgtitle('CV MSE vs Features for fixed Rules (only with ≥ 2 points)');
saveas(gcf, 'TSK_superconduct_results/CVMSE_vs_Features_allRules_filtered.png');
