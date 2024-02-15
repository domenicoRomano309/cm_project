clear;

lambda_values = [1e-10, 1e-5, 0.01, 1, 100, 1e5];
m_values = [3, 5, 7, 10];
max_iters = 100;


Stats = [];
Times = [];

for lambda = lambda_values
    [X_hat, y_hat] = Init(lambda);
    
    % effective minimum point (used to calculate the relative error)
    solution = linsolve(X_hat, y_hat);

    % initial approximation vector set to all zeros
    w_0 = zeros(size(solution));

    for m = m_values
        NewStats = L_BFGS(sparse(X_hat), y_hat, w_0, m, max_iters, solution, lambda);
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


