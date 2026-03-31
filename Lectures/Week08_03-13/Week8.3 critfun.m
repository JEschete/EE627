function dev = critfun(X,Y)

[b,dev] = glmfit(X,Y>0,'binomial');

