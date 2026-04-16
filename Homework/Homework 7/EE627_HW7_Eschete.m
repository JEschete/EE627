%% EE627 Homework 7 - Matrix Factorization
%  Jude Eschete
%
%  Tests the user-defined matrix_factorization function on the sample
%  rating matrix from the instructions.

clear all
close all

% R : Rating matrix (0 = missing entry)
R = [ 5 3 0 1;
      4 0 0 1;
      1 1 0 5;
      1 0 0 4;
      0 1 5 4];

[nRow, nCol] = size(R);

% K is the number of latent factors
K = 2;

rng(0);                 % reproducible initialization
P = rand(nRow, K);
Q = rand(K, nCol);

steps = 5000;
alpha = 0.0002;
beta  = 0.02;

fprintf('Original rating matrix R:\n');
disp(R);

fprintf('Initial P:\n');  disp(P);
fprintf('Initial Q:\n');  disp(Q);

[nP, nQ] = matrix_factorization(R, P, Q, K, steps, alpha, beta);

R_hat = nP * nQ;

fprintf('Learned P:\n');           disp(nP);
fprintf('Learned Q:\n');           disp(nQ);
fprintf('Reconstructed R_hat = P*Q:\n');  disp(R_hat);
fprintf('Original R:\n');          disp(R);

% Reconstruction error on observed entries only
mask = R > 0;
err_obs = sqrt(mean((R(mask) - R_hat(mask)).^2));
fprintf('RMSE on observed entries: %.6f\n', err_obs);

% Predictions for the missing entries (zeros in R)
fprintf('\nPredicted values for missing (zero) entries:\n');
[rowM, colM] = find(R == 0);
for t = 1:numel(rowM)
    fprintf('  R(%d,%d) = %.4f\n', rowM(t), colM(t), R_hat(rowM(t), colM(t)));
end
