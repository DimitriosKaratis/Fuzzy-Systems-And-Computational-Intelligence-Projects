%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 4
% Classification.pdf
%% Part 2: Epileptic Seizure Recognision Dataset 

clear; clc; close all;

tic

% Output directory for saving results
outDir = "TSK_epileptic_results";
if ~exist(outDir,'dir'); mkdir(outDir); end

fprintf("=== Epileptic Seizure Dataset - Class Dependent TSK ===\n");

%% Load dataset (excluding ID column)
T = readtable('epileptic_seizure_data.csv');
T(:,1) = [];  % remove ID column
data = table2array(T);

% Separate features and labels
X = data(:,1:end-1);   % features
y = data(:,end);       % class labels

%% Normalize input features to [0,1]
% Output (class label) is left unchanged
for i = 1:178
    data_min = min(X(:,i));
    data_max = max(X(:,i));
    X(:,i) = (X(:,i) - data_min) / (data_max - data_min);
end

% Dataset info
classes = unique(y);
numClasses = length(classes);
numFeaturesX = size(X,2);
fprintf("Dataset: %d samples, %d features, %d classes\n", size(X,1), numFeaturesX + 1, numClasses);

%% Stratified Split 60/20/20 
% Step 1: Train (60%) vs Rest (40%)
cv1 = cvpartition(y,'HoldOut',0.4,'Stratify',true);
X_train = X(training(cv1),:);
y_train = y(training(cv1));
X_rest  = X(test(cv1),:);
y_rest  = y(test(cv1));

% Step 2: Split Rest into Validation (20%) and Test (20%)
cv2 = cvpartition(y_rest,'HoldOut',0.5,'Stratify',true);
X_val = X_rest(training(cv2),:);
y_val = y_rest(training(cv2));
X_test = X_rest(test(cv2),:);
y_test = y_rest(test(cv2));

fprintf("Split sizes: Train=%d, Val=%d, Test=%d\n", ...
    size(X_train,1), size(X_val,1), size(X_test,1));

%% Feature Selection using ReliefF (on training set, K = 50)
[idx,weights] = relieff(X_train, y_train, 50);  % returns ranked feature indices

%% Grid search parameters
feature_candidates = [4 8 12 16 20];                      % number of features to test
radius_candidates  = [0.4 0.5 0.6 0.7 0.8];               % subclust radius
numFolds = 5;                                             % cross-validation folds


results = [];  
fprintf("\n--- Starting Grid Search ---\n");

%% Grid search with 5-fold CV on training set
for nf = feature_candidates
    sel_idx = idx(1:nf);          
    Xf = X_train(:,sel_idx);

    for ra = radius_candidates
        fprintf("Testing nf=%d, ra=%.2f\n", nf, ra);

        cv = cvpartition(y_train,'KFold',numFolds,'Stratify',true);
        cvErrs = zeros(cv.NumTestSets,1);
        numRules_total = 0;

        for i = 1:cv.NumTestSets
            Xtr = Xf(training(cv,i),:); ytr = y_train(training(cv,i));
            Xval_cv = Xf(test(cv,i),:); yval_cv = y_train(test(cv,i));

            %% Build class-dependent FIS
            fis = sugfis('Name','ClassDepFIS');

            % Add inputs
            for j=1:nf
                fis = addInput(fis,[0 1],'Name',['f' num2str(j)]);
            end

            % Add output
            fis = addOutput(fis,[min(classes) max(classes)],'Name','ClassOut');

            % Create rules using subclust for each class
            ruleList = [];
            for c = classes'
                Xc = Xtr(ytr==c,:);
                [centers,sigmas] = subclust(Xc,ra);
                for k=1:size(centers,1)
                    mfIdx = zeros(1,nf);
                    for j=1:nf
                        mfName = sprintf('c%d_r%d_in%d',c,k,j);
                        fis = addMF(fis,['f' num2str(j)],'gaussmf',...
                            [sigmas(j) centers(k,j)],'Name',mfName);
                        mfIdx(j) = length(fis.Inputs(j).MembershipFunctions);
                    end
                    % Add constant output MF
                    outName = sprintf('out_c%d_r%d',c,k);
                    fis = addMF(fis,'ClassOut','constant',c,'Name',outName);
                    outIdx = length(fis.Outputs(1).MembershipFunctions);
                    ruleList = [ruleList; [mfIdx outIdx 1 1]];
                end
            end
            fis = addRule(fis,ruleList);
            numRules_total = numRules_total + length(fis.Rules);

            %% Train FIS
            trnData = [Xtr ytr];
            valData = [Xval_cv yval_cv];
            opt = anfisOptions('InitialFIS',fis,'EpochNumber',250,...
                               'ValidationData',valData,...
                               'DisplayErrorValues',0,'DisplayANFISInformation',0);
            [~,~,~,~,chkErr] = anfis(trnData,opt);

            cvErrs(i) = min(chkErr);
        end

        % Store mean CV error and average rules
        meanErr = mean(cvErrs);
        numRules_avg = round(numRules_total/numFolds);
        fprintf("  -> Mean CV Error = %.4f, AvgRules=%d\n", meanErr, numRules_avg);

        results = [results; nf ra numRules_avg meanErr];
    end
end

%% Save grid search results
results_table = array2table(results, ...
    'VariableNames',{'NumFeatures','ClusterRadius','NumRules','MeanCV_Error'});
writetable(results_table, fullfile(outDir,'GridSearch_Results.csv'));

%% Select best parameters
[~,best_idx] = min(results(:,4));                           % select by minimum CV error
best_nf = results(best_idx,1);
best_ra = results(best_idx,2);
fprintf("\nBest parameters: nf=%d, ra=%.2f\n", best_nf, best_ra);

% Save selected features
selected_features = idx(1:best_nf);
writematrix(selected_features, fullfile(outDir,'Selected_Features.csv'));

%% Train final model using Train+Validation data
fprintf("\n--- Training Final Model ---\n");
X_trainVal = [X_train(:,selected_features); X_val(:,selected_features)];
y_trainVal = [y_train; y_val];

fis_final = sugfis('Name','FinalClassDep');

% Add inputs
for j=1:best_nf
    fis_final = addInput(fis_final,[0 1],'Name',['f' num2str(j)]);
end

% Add output
fis_final = addOutput(fis_final,[min(classes) max(classes)],'Name','ClassOut');

% Build rules using subclust
ruleList = [];
for c = classes'
    Xc = X_trainVal(y_trainVal==c,:);
    [centers,sigmas] = subclust(Xc,best_ra);
    for k=1:size(centers,1)
        mfIdx = zeros(1,best_nf);
        for j=1:best_nf
            mfName = sprintf('c%d_r%d_in%d',c,k,j);
            fis_final = addMF(fis_final,['f' num2str(j)],'gaussmf',...
                [sigmas(j) centers(k,j)],'Name',mfName);
            mfIdx(j) = length(fis_final.Inputs(j).MembershipFunctions);
        end
        % Add constant output MF
        outName = sprintf('out_c%d_r%d',c,k);
        fis_final = addMF(fis_final,'ClassOut','constant',c,'Name',outName);
        outIdx = length(fis_final.Outputs(1).MembershipFunctions);
        ruleList = [ruleList; [mfIdx outIdx 1 1]];
    end
end
fis_final = addRule(fis_final,ruleList);

%% Plot initial MFs for first 6 inputs
numToPlot = min(best_nf,6);
figure;
for i=1:numToPlot
    subplot(ceil(numToPlot/2),2,i);
    [x,mf] = plotmf(fis_final,'input',i);
    plot(x,mf);
    title(sprintf('Input %d Initial MFs',i));
end
sgtitle('Initial Membership Functions of FIS inputs (max 6)');
saveas(gcf,fullfile(outDir,'InitialFIS_MFs.png'));

%% Train final FIS using ANFIS
opt = anfisOptions('InitialFIS',fis_final,'EpochNumber',500,...
                   'ValidationData',[X_val(:,selected_features) y_val],...
                   'DisplayErrorValues',1);
[trnFIS,trainError,~,chkFIS,chkError] = anfis([X_trainVal y_trainVal],opt);

%% Evaluate on Test Set
y_pred = evalfis(chkFIS, X_test(:,selected_features));
y_pred_round = round(y_pred);                            % round to nearest integer class
y_pred_round(y_pred_round<min(classes))=min(classes);    % clip lower bound
y_pred_round(y_pred_round>max(classes))=max(classes);    % clip upper bound

C = confusionmat(y_test,y_pred_round);                   % confusion matrix
OA = sum(diag(C))/sum(C(:));                             % overall accuracy

%% Compute Per-Class Metrics
N = sum(C(:));
PA = diag(C)./sum(C,1)';                                 % Producer Accuracy (precision)
UA = diag(C)./sum(C,2);                                  % User Accuracy (recall)
p0 = OA;
pe = sum(sum(C,1).*sum(C,2))/N^2;
kappa = (p0-pe)/(1-pe);                                  % Cohen's Kappa

metrics_table = table(OA,kappa);
for c=1:numClasses
    metrics_table.(['PA_Class' num2str(c)]) = PA(c);
    metrics_table.(['UA_Class' num2str(c)]) = UA(c);
end
writetable(metrics_table, fullfile(outDir,'FinalModel_Metrics.csv'));

%% Grid Search Summary Printing 
fprintf("\n\n\n=== Grid Search Summary ===\n");
fprintf("Best Parameters Found:\n");
fprintf("  Number of Features (nf)      : %d\n", best_nf);
fprintf("  Cluster Radius (ra)          : %.2f\n", best_ra);

best_result = results_table(results_table.NumFeatures == best_nf & ...
                            results_table.ClusterRadius == best_ra, :);

fprintf("  Average Number of Rules      : %d\n", best_result.NumRules);
fprintf("  Mean CV Error                : %.4f\n", best_result.MeanCV_Error);

fprintf("\nFinal Model Test Metrics:\n");
fprintf("  Overall Accuracy (OA)        : %.3f\n", OA);
fprintf("  Cohen's Kappa                : %.3f\n", kappa);
for c = 1:numClasses
    fprintf("  PA Class %d                   : %.3f\n", c, metrics_table.(['PA_Class' num2str(c)]));
    fprintf("  UA Class %d                   : %.3f\n", c, metrics_table.(['UA_Class' num2str(c)]));
end

fprintf("\nResults saved in folder: %s\n", outDir);

%% Class distribution table
dist_table = table(classes, ...
    histcounts(y_train, [classes; max(classes)+1])', ...
    histcounts(y_val,   [classes; max(classes)+1])', ...
    histcounts(y_test,  [classes; max(classes)+1])', ...
    'VariableNames', {'Class','Training','Validation','Testing'});

writetable(dist_table, fullfile(outDir,'Class_Distribution.csv'));


%% Plot final MFs for first 6 inputs
figure;
for i=1:numToPlot
    subplot(ceil(numToPlot/2),2,i);
    [x,mf] = plotmf(chkFIS,'input',i);
    plot(x,mf);
    title(sprintf('Input %d Final MFs',i));
end
sgtitle('Final Membership Functions of FIS inputs (max 6)');
saveas(gcf,fullfile(outDir,'FinalFIS_MFs.png'));

%% Plot Learning Curves
[minValError,bestEpoch] = min(chkError);
figure;
plot(1:length(trainError),trainError,'b','LineWidth',1.5); hold on;
plot(1:length(chkError),chkError,'r','LineWidth',1.5);
plot(bestEpoch,minValError,'ko','MarkerSize',8,'MarkerFaceColor','g');
xlabel('Epoch'); ylabel('Error');
legend('Training','Validation','Best chkFIS');
title('Learning Curve (Final Model)');
grid on;
saveas(gcf,fullfile(outDir,'Learning_Curve.png'));

%% Confusion Matrix Heatmap
figure;
confusionchart(y_test,y_pred_round);
title('Confusion Matrix - Test Set');
saveas(gcf,fullfile(outDir,'ConfusionMatrix.png'));

% Normalized Confusion Matrix
figure;
confusionchart(y_test,y_pred_round, 'Normalization','row-normalized');
title('Normalized Confusion Matrix - Test Set');
saveas(gcf, fullfile(outDir,'ConfusionMatrix_Normalized.png'));



%% Plot True vs Predicted Subplots (offset version)
nSubplots = 6;
nSamples = length(y_test);
chunkSize = ceil(nSamples / nSubplots);

figure;
for i = 1:nSubplots
    idx_start = (i-1)*chunkSize + 1;
    idx_end = min(i*chunkSize, nSamples);
    idx_range = idx_start:idx_end;

    subplot(3,2,i);
    scatter(idx_range, y_test(idx_range), 40, 'b', 'filled'); hold on;
    scatter(idx_range, y_pred_round(idx_range)+0.05, 25, 'r', 'filled'); % offset for visibility
    ylabel('Class');
    title(sprintf('Samples %d - %d', idx_start, idx_end));
    grid on;
    if i > 4
        xlabel('Sample Index');
    end
    if i == 1
        legend('True','Predicted (offset)');
    end
end
sgtitle('True vs Predicted Classes (Test Set) - Offset Version');
saveas(gcf, fullfile(outDir,'True_vs_Predicted_Offset_Subplots.png'));

%% Plot Continuous and Discrete Prediction Errors
errors_cont = y_test - y_pred;
errors_round = y_test - y_pred_round;

figure;
plot(errors_cont,'b','LineWidth',1.5); hold on;
xlabel('Sample'); ylabel('Error');
title('Continuous FIS Prediction Errors (Test Set)');
grid on;
saveas(gcf,fullfile(outDir,'Prediction_Errors_Continuous.png'));

figure;
plot(errors_round,'g','LineWidth',1.5); hold on;
xlabel('Sample'); ylabel('Error');
title('Discrete FIS Prediction Errors (Test Set)');
grid on;
saveas(gcf,fullfile(outDir,'Prediction_Errors_Discrete.png'));

%% 3D Plot of Grid Search Results
figure;
scatter3(results(:,1),results(:,3),results(:,4),80,results(:,4),'filled');
colormap jet; colorbar;
xlabel('NumFeatures'); ylabel('NumRules'); zlabel('Mean CV Error');
title('Grid Search Results: Features vs Rules vs Error');
grid on; view(45,30);
saveas(gcf,fullfile(outDir,'GridSearch_3DPlot.png'));

%% t-SNE Visualization 

% t-SNE Visualization of All Features
Y_tsne_all = tsne(X);
figure;
gscatter(Y_tsne_all(:,1), Y_tsne_all(:,2), y);
xlabel('t-SNE Dim 1'); ylabel('t-SNE Dim 2');
title('t-SNE with All Features');
legend('Class 1','Class 2','Class 3','Class 4','Class 5','Location','bestoutside');
grid on;
saveas(gcf,fullfile(outDir,'tSNE_All_Features.png'));

% t-SNE Visualization of Selected Features
X_selected = X(:, selected_features);
y_selected = y;
Y_tsne = tsne(X_selected);
figure;
gscatter(Y_tsne(:,1), Y_tsne(:,2), y_selected);
xlabel('t-SNE Dim 1'); ylabel('t-SNE Dim 2');
title('t-SNE Projection of Selected Features');
legend('Class 1','Class 2','Class 3','Class 4','Class 5');
grid on;
saveas(gcf,fullfile(outDir,'tSNE_Selected_Features.png'));



%% Grid Search Visualization 

% CV Error vs Number of Rules (for fixed NumFeatures)
feature_list = unique(results_table.NumFeatures);
nFeat = length(feature_list);

figure;
for i = 1:nFeat
    nf = feature_list(i);
    idx = results_table.NumFeatures == nf;
    numRules = results_table.NumRules(idx);
    cvErr = results_table.MeanCV_Error(idx);

    [numRules_sorted, sortIdx] = sort(numRules);
    cvErr_sorted = cvErr(sortIdx);

    subplot(ceil(sqrt(nFeat)), ceil(sqrt(nFeat)), i);
    plot(numRules_sorted, cvErr_sorted, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Rules'); ylabel('Mean CV Error');
    title(sprintf('Features = %d', nf));
    grid on;
end
sgtitle('CV Error vs Rules for fixed Features');
saveas(gcf, fullfile(outDir,'CVMSE_vs_Rules_allFeatures.png'));



% CV Error vs Number of Features (for fixed NumRules)
rules_list = unique(results_table.NumRules);
nRules = length(rules_list);

% Only consider rules with >=2 feature values
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
    idx = results_table.NumRules == nr;
    numFeatures = results_table.NumFeatures(idx);
    cvErr = results_table.MeanCV_Error(idx);

    [numFeatures_sorted, sortIdx] = sort(numFeatures);
    cvErr_sorted = cvErr(sortIdx);

    subplot(ceil(sqrt(nValid)), ceil(sqrt(nValid)), i);
    plot(numFeatures_sorted, cvErr_sorted, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Features'); ylabel('Mean CV Error');
    title(sprintf('Rules = %d', nr));
    grid on;
end
sgtitle('CV Error vs Features for fixed Rules');
saveas(gcf, fullfile(outDir,'CVMSE_vs_Features_allRules_filtered.png'));

toc