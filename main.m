clear;

lambda_values = logspace(-10, 5, 16);
m_values = [3, 4, 5, 6, 7, 8, 9, 10];
max_iters = 100;


Stats = [];
Times = [];

for lambda = lambda_values
    fprintf('\nlambda = %d\n',lambda);

    [X_hat, y_hat] = Init(lambda);

    sv_Xhat=svd(X_hat);

    % effective minimum point (used to calculate the relative error)
    solution = linsolve(X_hat, y_hat);

    % initial approximation vector set to all zeros
    w_0 = zeros(size(solution));


    cond_num_Xhat = cond(X_hat);

    for m = m_values
        
        fprintf('m = %d\n',m);
        NewStats = L_BFGS(sparse(X_hat), y_hat, w_0, m, max_iters, solution, ...
            lambda, sv_Xhat, cond_num_Xhat);
        Stats = [Stats; NewStats];
        f = @() L_BFGS_chrono(X_hat, y_hat, w_0, m, max_iters);
        new_row = [lambda m timeit(f)];
        Times = [Times; new_row];

    end
    
end

writematrix(Stats,'statistics.csv')
writematrix(Times,'times.csv')




function [X_hat, y_hat] = Init(lambda)
seed = 42;
rng(seed);

X = table2array(readtable("./ML-CUP22-TR.csv"));
X = X(:, 2:end);


X_hat = [X'; lambda * eye(size(X,1))];

y = randn(size(X,2), 1); 
y_hat = [y; zeros(size(X,1), 1)];
end


