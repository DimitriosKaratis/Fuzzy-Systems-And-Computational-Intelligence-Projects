%% Fuzzy Systems
% KARATIS DIMITRIOS 10775 - Assignment 4
% Classification.pdf
%% Part 1: Haberman's Survival Dataset 
% TSK Model 3 -> class-dependent + radius = 0.2 ("small")

clear all; close all; clc;

% Start timer to measure execution time
tic

%% Load dataset (Haberman's Survival)
% Dataset format: 3 inputs (Age, Year, Positive Nodes), 1 output (Survival class)
data = load('haberman.data'); 

%% Plot histograms of the input features
figure;
featureNames = {'Age', 'Year of Operation', 'Positive Nodes'};
numFeatures = 3;

for i = 1:numFeatures
    subplot(1, numFeatures, i);
    histogram(data(:,i), 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
    xlabel(featureNames{i});
    ylabel('Count');
    grid on;
end
sgtitle('Input Features Distribution - Haberman Dataset');
saveas(gcf, 'InputFeaturesDistribution.png');

%% Plot features vs sample index (uncolored)
figure;
for i = 1:numFeatures
    subplot(3,1,i);
    plot(1:size(data,1), data(:,i), 'o-', 'LineWidth', 1.2, 'MarkerSize',4);
    xlabel('Sample Index');
    ylabel(featureNames{i});
    title([featureNames{i}, ' vs Sample Index']);
    grid on;
end
sgtitle('Input Features vs Sample Index');
saveas(gcf, 'InputFeaturesVsIndex.png');

%% Plot features vs sample index (colored by class)
figure;
colors = [0 0 1; 1 0 0];  % Blue = class 1, Red = class 2
for i = 1:numFeatures
    subplot(3,1,i); hold on;
    for c = 1:2
        idx = data(:,4) == c;
        plot(find(idx), data(idx,i), 'o', ...
             'Color', colors(c,:), ...
             'MarkerFaceColor', colors(c,:), ...
             'MarkerSize',5);
    end
    xlabel('Sample Index');
    ylabel(featureNames{i});
    title([featureNames{i}, ' vs Sample Index']);
    grid on;
end
sgtitle('Input Features vs Sample Index (Colored by Class)');
legend({'Class 1','Class 2'}, 'Location','best');
saveas(gcf, 'InputFeaturesColoredByClass.png');

%% Print class distribution
classes = unique(data(:,4));
for c = 1:length(classes)
    fprintf('Class %d count = %d\n', classes(c), sum(data(:,4)==classes(c)));
end

%% Normalize input features to [0,1]
% Output (class label) is kept unchanged
for i = 1:numFeatures
    data_min = min(data(:,i));
    data_max = max(data(:,i));
    data(:,i) = (data(:,i) - data_min) / (data_max - data_min);
end

% Output classes are {1,2}
classes = unique(data(:,4));
fprintf('Classes present: %s\n', mat2str(classes));

%% Experiment parameters
numRuns = 30;    % Number of independent runs for statistics
radius = 0.2;    % Cluster radius for subtractive clustering
allMetrics = cell(numRuns,1);

fprintf('--- Class-Dependent TSK Classification --- Model 3 ---\n');

for run = 1:numRuns
    fprintf('Run %d/%d\n', run, numRuns);
    rng(run); % Set random seed for reproducibility

    %% Stratified split into training, validation, and test sets (60/20/20)
    cv1 = cvpartition(data(:,4), 'HoldOut', 0.4, 'Stratify', true); 
    trainingData = data(training(cv1), :);
    restData     = data(test(cv1), :);

    cv2 = cvpartition(restData(:,4), 'HoldOut', 0.5, 'Stratify', true);
    validationData = restData(training(cv2), :);
    testData       = restData(test(cv2), :);

    %% Class-dependent subtractive clustering
    % Each class has its own clusters and parameters
    centers = cell(length(classes),1);
    sigmas  = cell(length(classes),1);
    numRulesPerClass = zeros(length(classes),1);

    for c = 1:length(classes)
        idx = trainingData(:,4) == classes(c);
        [centers{c}, sigmas{c}] = subclust(trainingData(idx,1:3), radius);
        numRulesPerClass(c) = size(centers{c},1);
    end
    numRules = sum(numRulesPerClass);

    %% Build Sugeno FIS
    fis = sugfis('Name','ClassDependentFIS');

    % Add inputs
    for i=1:numFeatures
        fis = addInput(fis,[0 1],'Name',featureNames{i});
    end

    % Add output
    fis = addOutput(fis,[min(classes) max(classes)],'Name','ClassOut');

    % Add membership functions (MFs) and rules per class
    ruleList = [];
    ruleIndex = 1;

    for c = 1:length(classes)
        for j = 1:numRulesPerClass(c)
            mfIdx = zeros(1,numFeatures);
            for i = 1:numFeatures
                mfName = sprintf('c%d_r%d_in%d',c,j,i);
                fis = addMF(fis,featureNames{i},'gaussmf',...
                    [sigmas{c}(i) centers{c}(j,i)],'Name',mfName);
                mfIdx(i) = length(fis.Inputs(i).MembershipFunctions);
            end
            % Add constant output MF
            outName = sprintf('out_c%d_r%d',c,j);
            fis = addMF(fis,'ClassOut','constant',classes(c),'Name',outName);
            outIdx = length(fis.Outputs(1).MembershipFunctions);

            % Create fuzzy rule
            ruleList = [ruleList; [mfIdx outIdx 1 1]];
            ruleIndex = ruleIndex + 1;
        end
    end

    fis = addRule(fis,ruleList);

    %% Train FIS using ANFIS (hybrid learning)
    opt = anfisOptions('InitialFIS',fis,'EpochNumber',100,...
                       'ValidationData',validationData,...
                       'DisplayANFISInformation',0,...
                       'DisplayErrorValues',0,...
                       'DisplayStepSize',0);
    [trnFIS,trnError,stepSize,chkFIS,chkError] = anfis(trainingData,opt);

    %% Evaluate FIS on test set
    y_pred = evalfis(chkFIS, testData(:,1:3));

    % Round outputs to nearest class
    y_pred_round = round(y_pred);
    y_pred_round(y_pred_round<min(classes))=min(classes);
    y_pred_round(y_pred_round>max(classes))=max(classes);

    y_true = testData(:,4);

    % Confusion matrix
    C = confusionmat(y_true,y_pred_round);

    % Performance metrics
    N = sum(C(:));
    OA = sum(diag(C))/N;        % Overall Accuracy
    PA = diag(C)./sum(C,1)';    % Producer's Accuracy (per class)
    UA = diag(C)./sum(C,2);     % User's Accuracy (per class)
    
    % Kappa coefficient
    total = sum(C(:));
    p0 = sum(diag(C))/total;
    pe = sum(sum(C,1).*sum(C,2))/total^2;
    kappa = (p0-pe)/(1-pe);

    % Store metrics
    allMetrics{run} = struct('C',C,'OA',OA,'PA',PA,'UA',UA,'kappa',kappa,...
                             'trnError',trnError,'chkError',chkError,...
                             'fisInit',fis,'fisFinal',chkFIS,...
                             'y_true',y_true,'y_pred',y_pred_round);
end

%% Compute statistics across runs
OAs = cellfun(@(m) m.OA, allMetrics);
Kappas = cellfun(@(m) m.kappa, allMetrics);
PAs = cell2mat(cellfun(@(m) m.PA(:)', allMetrics,'UniformOutput',false));
UAs = cell2mat(cellfun(@(m) m.UA(:)', allMetrics,'UniformOutput',false));

meanOA = mean(OAs); stdOA = std(OAs);
meanKappa = mean(Kappas); stdKappa = std(Kappas);
meanPA = mean(PAs,1); stdPA = std(PAs,0,1);
meanUA = mean(UAs,1); stdUA = std(UAs,0,1);

fprintf('\n==== Summary over %d runs ====\n', numRuns);
fprintf('OA:    mean = %.3f, std = %.3f\n', meanOA, stdOA);
fprintf('Kappa: mean = %.3f, std = %.3f\n', meanKappa, stdKappa);
for c = 1:length(classes)
    fprintf('PA (Class %d): mean = %.3f, std = %.3f\n', classes(c), meanPA(c), stdPA(c));
    fprintf('UA (Class %d): mean = %.3f, std = %.3f\n', classes(c), meanUA(c), stdUA(c));
end

%% Find representative run (closest OA to mean OA)
[~, repIdx] = min(abs(OAs - meanOA));
repRun = allMetrics{repIdx};
fprintf('\nRepresentative run: %d\n', repIdx);
fprintf('OA = %.3f, Kappa = %.3f\n', repRun.OA, repRun.kappa);

%% Class distribution across splits (Training, Validation, Test)
classes = unique(data(:,4));
numClasses = length(classes);
freqTable = zeros(numClasses, 3);
for c = 1:numClasses
    freqTable(c,1) = sum(trainingData(:,4) == classes(c));
    freqTable(c,2) = sum(validationData(:,4) == classes(c));
    freqTable(c,3) = sum(testData(:,4) == classes(c));
end
T = array2table(freqTable, 'VariableNames', {'Training','Validation','Testing'}, ...
 'RowNames', strcat('Class_', string(classes)));
disp(T);

%% Plotting for representative run
fisBefore = repRun.fisInit;
fisAfter  = repRun.fisFinal;

% Membership functions before training
figure;
for i=1:3
    subplot(2,2,i);
    [x,mf] = plotmf(fisBefore,'input',i);
    plot(x,mf);
    title(['Input ' num2str(i) ' MFs (Before)']);
end
sgtitle('Membership Functions Before Training');
saveas(gcf, fullfile('TSK_model_3_imgs','3_MFsBeforeTraining.png'));

% Membership functions after training
figure;
for i=1:3
    subplot(2,2,i);
    [x,mf] = plotmf(fisAfter,'input',i);
    plot(x,mf);
    title(['Input ' num2str(i) ' MFs (After)']);
end
sgtitle('Membership Functions After Training');
saveas(gcf, fullfile('TSK_model_3_imgs','3_MFsAfterTraining.png'));

% Confusion matrix heatmap
figure;
confusionchart(repRun.y_true, repRun.y_pred);
title('Confusion Matrix (Representative Run)');
saveas(gcf, fullfile('TSK_model_3_imgs','3_ConfusionMatrix.png'));

% Prediction errors plot
figure;
errors = repRun.y_true - repRun.y_pred;
stem(errors,'filled'); grid on;
xlabel('Sample Index'); ylabel('Error (True - Pred)');
title('Prediction Errors (Representative Run)');
saveas(gcf, fullfile('TSK_model_3_imgs','3_PredictionErrors.png'));

% Learning curve (training vs validation error)
figure;
plot(1:length(repRun.trnError), repRun.trnError,'b','LineWidth',1.5); hold on;
plot(1:length(repRun.chkError), repRun.chkError,'r','LineWidth',1.5);
[~, minValIdx] = min(repRun.chkError);
plot(minValIdx, repRun.chkError(minValIdx),'go','MarkerSize',8,'MarkerFaceColor','g');
xlabel('Epochs'); ylabel('Error');
legend('Training','Validation','Best chkFIS');
title('Learning Curve (Representative Run)'); grid on;
saveas(gcf, fullfile('TSK_model_3_imgs','3_LearningCurve.png'));

% Display FIS parameters
fprintf('\nFIS Parameters for representative run:\n');
fprintf('Cluster radius: %.2f\n', radius);
fprintf('Number of rules: %d\n', length(fisAfter.Rules));

% Stop timer
toc
