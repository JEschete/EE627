clear all
close all

R=[ 5 3 0 1; 
    4 0 0 1; 
    1 1 0 5; 
    1 0 0 4; 
    0 1 5 4];
[nRow, nCol]=size(R);
K=2;
P=rand(nRow, K);
Q=rand(K, nCol);

steps=5000;
alpha=0.0002;
beta=0.02;


[nP, nQ]=matrix_factorization(R,P,Q,K, steps, alpha, beta);
[nP*nQ]
[R]



R=[ 4 3 0 1 2; 
    5 0 0 1 0; 
    1 2 1 5 4; 
    1 0 0 4 0; 
    0 1 5 4 0;
    5 5 0 0 1];
[nRow, nCol]=size(R);
K=3;
P=rand(nRow, K);
Q=rand(K, nCol);

steps=10000;
alpha=0.0002;
beta=0.02;


[nP, nQ]=matrix_factorization(R,P,Q,K, steps, alpha, beta);
[nP*nQ]
[R]


