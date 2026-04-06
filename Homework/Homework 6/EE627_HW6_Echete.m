%% EE627 Homework 6 - Eschete
% Feature Engineering and Feature Selection with Logistic Regression
clear; close all; clc;

%% Load Data
load('dataSet_2.mat');
P = size(predictor_train, 2);
fprintf('Training set: %d samples, %d features\n', size(predictor_train,1), P);
fprintf('Test set:     %d samples, %d features\n', size(predictor_test,1), P);

%% Step 1: Compute Daily Change Rate
cr_train = (predictor_train(2:end,:) - predictor_train(1:end-1,:)) ./ predictor_train(1:end-1,:);
cr_test  = (predictor_test(2:end,:)  - predictor_test(1:end-1,:))  ./ predictor_test(1:end-1,:);
cr_train(isnan(cr_train)) = 0;  cr_train(isinf(cr_train)) = 0;
cr_test(isnan(cr_test))   = 0;  cr_test(isinf(cr_test))   = 0;

% Drop first row to align
X_train_orig = predictor_train(2:end, :);
X_test_orig  = predictor_test(2:end, :);
y_train = response_train(2:end);
y_test  = response_test(2:end);

% Combined: original + change rate
X_train_comb = [X_train_orig, cr_train];
X_test_comb  = [X_test_orig,  cr_test];
P2 = size(X_train_comb, 2);
fprintf('Combined: %d train / %d test, %d features (%d orig + %d CR)\n', ...
    length(y_train), length(y_test), P2, P, P2-P);

%% Step 2a: Baseline - Original Predictors Only
fprintf('\n=== Baseline: Original Predictors Only ===\n');

factors_orig = glmfit(X_train_orig, y_train, 'binomial');
prob_tr_orig = glmval(factors_orig, X_train_orig, 'logit');
prob_te_orig = glmval(factors_orig, X_test_orig, 'logit');

[fpr_tr, tpr_tr, ~, auc_tr] = perfcurve(y_train, prob_tr_orig, 1);
[fpr_te, tpr_te, ~, auc_te] = perfcurve(y_test, prob_te_orig, 1);
fprintf('Training AUC = %.4f\n', auc_tr);
fprintf('Test AUC     = %.4f\n', auc_te);

figure(1);
plot(fpr_tr, tpr_tr, 'b-', 'LineWidth', 2); hold on;
plot(fpr_te, tpr_te, 'r-', 'LineWidth', 2);
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC - Original Predictors Only');
legend(sprintf('Train (AUC=%.4f)', auc_tr), ...
       sprintf('Test  (AUC=%.4f)', auc_te), ...
       'Random', 'Location', 'southeast');
grid on; hold off;

%% Step 2b: Combined Predictors (Original + Change Rate)
fprintf('\n=== Combined Predictors (Original + Change Rate) ===\n');

factors_comb = glmfit(X_train_comb, y_train, 'binomial');
prob_tr_comb = glmval(factors_comb, X_train_comb, 'logit');
prob_te_comb = glmval(factors_comb, X_test_comb, 'logit');

[fpr_tr2, tpr_tr2, ~, auc_tr2] = perfcurve(y_train, prob_tr_comb, 1);
[fpr_te2, tpr_te2, ~, auc_te2] = perfcurve(y_test, prob_te_comb, 1);
fprintf('Training AUC = %.4f\n', auc_tr2);
fprintf('Test AUC     = %.4f\n', auc_te2);

figure(2);
plot(fpr_tr2, tpr_tr2, 'b-', 'LineWidth', 2); hold on;
plot(fpr_te2, tpr_te2, 'r-', 'LineWidth', 2);
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC - Combined Predictors (Original + Change Rate)');
legend(sprintf('Train (AUC=%.4f)', auc_tr2), ...
       sprintf('Test  (AUC=%.4f)', auc_te2), ...
       'Random', 'Location', 'southeast');
grid on; hold off;

%% Step 3: Feature Selection on Combined Predictors
fprintf('\n=== Feature Selection (Sequential Forward) ===\n');
[SelectedFeatureInd] = featureSelection(X_train_comb, y_train);

selectedCols = find(SelectedFeatureInd);
numSelected = length(selectedCols);
fprintf('Number of features selected: %d\n', numSelected);
fprintf('Selected feature indices: ');
fprintf('%d ', selectedCols);
fprintf('\n');

% Identify which selected features are original vs change rate
origSelected = selectedCols(selectedCols <= P);
crSelected   = selectedCols(selectedCols > P) - P;
fprintf('  From original predictors: %d features (indices: ', length(origSelected));
fprintf('%d ', origSelected); fprintf(')\n');
fprintf('  From change rate predictors: %d features (indices: ', length(crSelected));
fprintf('%d ', crSelected); fprintf(')\n');

%% Step 4: Evaluate Different Numbers of Top Features
fprintf('\n=== AUC vs Number of Top Features ===\n');

% Rank features by individual deviance (lower = better)
deviances = zeros(1, P2);
for j = 1:P2
    deviances(j) = critfun(X_train_comb(:,j), y_train);
end
[~, featureRank] = sort(deviances, 'ascend');

topKs = [5, 10, 15, 20, 25, 30, 50];
topKs = topKs(topKs <= P2);
AUC_train_topK = zeros(size(topKs));
AUC_test_topK  = zeros(size(topKs));
legendStr = cell(1, length(topKs));

fprintf('%-10s %-15s %-15s\n', 'Top K', 'Train AUC', 'Test AUC');
fprintf('%s\n', repmat('-', 1, 40));

for i = 1:length(topKs)
    k = topKs(i);
    cols = featureRank(1:k);

    factors_k = glmfit(X_train_comb(:, cols), y_train, 'binomial');
    prob_tr_k = glmval(factors_k, X_train_comb(:, cols), 'logit');
    prob_te_k = glmval(factors_k, X_test_comb(:, cols), 'logit');

    [~, ~, ~, AUC_train_topK(i)] = perfcurve(y_train, prob_tr_k, 1);
    [~, ~, ~, AUC_test_topK(i)]  = perfcurve(y_test, prob_te_k, 1);

    fprintf('%-10d %-15.4f %-15.4f\n', k, AUC_train_topK(i), AUC_test_topK(i));
end

figure(3);
plot(topKs, AUC_train_topK, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(topKs, AUC_test_topK, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Top Features'); ylabel('AUC');
title('AUC vs Number of Selected Features');
legend('Training AUC', 'Test AUC', 'Location', 'best');
grid on; hold off;

%% Step 5: ROC Curves for Selected Feature Subsets
figure(4);
colors = lines(length(topKs));
for i = 1:length(topKs)
    k = topKs(i);
    cols = featureRank(1:k);

    factors_k = glmfit(X_train_comb(:, cols), y_train, 'binomial');
    prob_te_k = glmval(factors_k, X_test_comb(:, cols), 'logit');

    [Xr, Yr, ~, auc_val] = perfcurve(y_test, prob_te_k, 1);
    plot(Xr, Yr, 'Color', colors(i,:), 'LineWidth', 1.5); hold on;
    legendStr{i} = sprintf('Top %d (AUC=%.4f)', k, auc_val);
end
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Test ROC Curves for Different Feature Subsets');
legend([legendStr, {'Random'}], 'Location', 'southeast');
grid on; hold off;

fprintf('\nDone.\n');
