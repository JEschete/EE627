function [P, Q] = matrix_factorization(R, P, Q, K, steps, alpha, beta)
% MATRIX_FACTORIZATION  Non-negative matrix factorization via stochastic
% gradient descent with L2 regularization.
%
%   [P, Q] = matrix_factorization(R, P, Q, K, steps, alpha, beta)
%
%   Inputs:
%     R     - nRow x nCol rating matrix (0 entries are treated as missing)
%     P     - nRow x K initial user-factor matrix
%     Q     - K x nCol initial item-factor matrix
%     K     - number of latent factors
%     steps - maximum number of SGD iterations
%     alpha - learning rate
%     beta  - L2 regularization coefficient
%
%   Outputs:
%     P, Q  - updated factor matrices such that R ~ P*Q for observed entries
%
%   The algorithm minimizes the regularized squared error:
%       E = sum_{(i,j) in obs} ( (R(i,j) - P(i,:)*Q(:,j))^2
%                              + (beta/2)*( ||P(i,:)||^2 + ||Q(:,j)||^2 ) )
%   updating each P(i,k) and Q(k,j) with the SGD rules:
%       P(i,k) <- P(i,k) + alpha*(2*e_ij*Q(k,j) - beta*P(i,k))
%       Q(k,j) <- Q(k,j) + alpha*(2*e_ij*P(i,k) - beta*Q(k,j))
%   Iteration stops early when total regularized error drops below 0.001.

[nRow, nCol] = size(R);

for step = 1:steps
    for i = 1:nRow
        for j = 1:nCol
            if R(i, j) > 0
                eij = R(i, j) - P(i, :) * Q(:, j);
                for k = 1:K
                    P_ik = P(i, k);
                    Q_kj = Q(k, j);
                    P(i, k) = P_ik + alpha * (2 * eij * Q_kj - beta * P_ik);
                    Q(k, j) = Q_kj + alpha * (2 * eij * P_ik - beta * Q_kj);
                end
            end
        end
    end

    e = 0;
    for i = 1:nRow
        for j = 1:nCol
            if R(i, j) > 0
                e = e + (R(i, j) - P(i, :) * Q(:, j))^2;
                for k = 1:K
                    e = e + (beta / 2) * (P(i, k)^2 + Q(k, j)^2);
                end
            end
        end
    end

    if e < 0.001
        break;
    end
end
end
