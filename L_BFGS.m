% Algorithm 7.5 in "Numerical optimization" (Nocedal, Wright) page 179
function Statistics = L_BFGS(X_hat, y_hat, x_0, m, max_iters, solution, lambda)
%
% Description: implementation by scratch of L-BFGS method in order to 
%   solve the  following LLS problem:   min_w||X_hat*w - y_hat||
%
% Input:
%   X_hat
%   y_hat 
%   x_0: initial approximation of the solution
%   m: max number of (s_k, y_k) couples saved in memory
%   max_iters
%   solution: effective minimum point (used to calculate the relative error)
%   lambda: used to construct the X_hat matrix, used in this function just
%       to write the statistics
%
% Output:
%   Statistics: matrix which contains each iteration's statistics (number 
%       of the iteration, gradient norm, relative error norm) and also the
%       values of the current hyperparameters (lambda, m)
%

Statistics = zeros(max_iters, 5);

Q_qe = 2* (X_hat'* X_hat); %Q of the quadratic equation
sz = size(Q_qe);

H_0 = eye(sz);

% definition of the gradient function (grad. of our function g)
grad_g = @(w) w'* (Q_qe) - 2*y_hat'*X_hat;

H0_k = H_0;
x_k = x_0;
grad_f_k = grad_g(x_0)';

S=zeros(size(x_k,1),m);
Y=zeros(size(x_k,1),m);
rho=zeros(1,m);

k=0;
UpdateStats();


while (norm(grad_f_k) > 1e-10) && (k < max_iters)
    k = k + 1;
    
    p_k = TwoLoopRecursion();
    
    % step size calculated by exact line search
    step_size_k = - (grad_f_k' * p_k) / (p_k' *Q_qe* p_k);
    x_kplus1 = x_k + step_size_k*p_k;
    grad_f_kplus1 = grad_g(x_kplus1)';
    
    S(:, Pos(k,m)) = x_kplus1 - x_k;
    Y(:, Pos(k,m)) = grad_f_kplus1 - grad_f_k;
    rho(Pos(k,m)) = 1/( Y(:, Pos(k,m))' * S(:, Pos(k,m)) );


    x_k = x_kplus1;
    grad_f_k = grad_f_kplus1; 

    UpdateStats();

   

    gamma_k = (S(:,Pos(k,m))' * Y(:,Pos(k,m))) / (Y(:,Pos(k,m))' * Y(:,Pos(k,m)));
    H0_k = gamma_k * eye(sz);
end

Statistics = Statistics(1:(k+1), :);

    
    
    % Algorithm 7.4 in "Numerical optimization" (Nocedal, Wright) page 178
    function r = TwoLoopRecursion()
        q = grad_f_k;
        alpha = zeros(1,m);

        if k <= m 
            m_star = k-1;
        else
            m_star = m;
        end

        for index = 1:m_star
            i = Pos((k - index),m);

            % alpha is an array of scalars
            alpha(i) = rho(i) * S(:,i)' * q; 
            q = q - alpha(i)*Y(:,i);
        end

        r = H0_k * q;

        for index = m_star:-1:1
            i = Pos((k - index),m);
            
            beta = rho(i) * Y(:,i)' * r;
            r = r + S(:,i)*(alpha(i)-beta);    
        end
        % r at the end will be H_k * grad_f_k
        % (H_k is intended as the k-th approximation of the inverse of
        % the Hessian)
    end


    function UpdateStats()
        Statistics(k+1, 1) = lambda;
        Statistics(k+1, 2) = m;
        Statistics(k+1, 3) = k;
    
        rel_error = norm(x_k - solution)/norm(solution);
        Statistics(k+1, 4) = rel_error;

        Statistics(k+1, 5) = norm(grad_f_k);

    end


    function p = Pos(a,b)
        if mod(a,b) == 0
            p = b;
        else
            p = mod(a,b);
        end
    end

end

