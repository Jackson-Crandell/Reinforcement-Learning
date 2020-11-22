%% Finite Differencing
% Dynamics
global A; 
global B; 
global Q; 
global R; 

global Horizon; 
global dt; 
global rollouts; 

global x0; 

global sigma; 

A = [0.4];
B = [0.9];
Q = [0.01];
R = [0.001];
dt = 0.001;

x0 = 1;   % initialize initial state
Horizon = 300;        %N
rollouts = 100; %M


% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R);

% [K_LQR,S,E] = lqr(A,B,Q,R);

[~,x_lqr] = trajectory_cost(K_LQR);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(1,Horizon-1);
r = zeros(1,Horizon);

delta_theta = zeros(rollouts,1);
delta_J = zeros(rollouts,1);

sigma = 0.01;    % noise level 

theta = 0.0;      % initialize theta

% Line search parameters
alpha = 0.3;    % learning rate 
c = 0.3 
p = 0.9


J = expected_cost(theta);   % Set the initial cost 

grad_J = 0; 
eps = 1e-4

iter = 0;       % number of iterations

converged_count = 0; 

while converged_count < 10
    
    iter = iter + 1;
     
    expected_reward = 0; 
    
    % Every itetation compute m rollouts
    for m = 1:rollouts
     
        % Perturb the parameter in the policy 
        delta_theta(m,1) = (2*randn - 1)*sigma;  % randomn number bw [-1,1]*sigma
        new_theta = theta(1,iter) + delta_theta(m,1);  
        
        % total reward of the current rollout
        [new_J,~] = trajectory_cost(new_theta);
        
        % store the difference in the reward between current rollout and 
        % reward at previous iteration
        delta_J(m,1) = new_J - J(iter);
        
    end
    
    % Compute the gradient using the Finite Difference Method
    grad_J(1,iter) = (inv(delta_theta' * delta_theta)) * delta_theta' * delta_J; 
    
    % Update the parameter in the policy 
    % backtracking line search
    % alpha = line_search(theta,grad_J,alpha,c,p)
    alpha = 1/sqrt(iter)
    theta(1,iter+1) = theta(1,iter) + alpha*grad_J(1,iter);

    J(1,iter+1) = expected_cost(theta(1,iter));
   
    grad_converged = abs(grad_J(1,iter))
    %grad_converged = abs(J(1,iter+1) - J(1,iter))

    if grad_converged < eps
        converged_count = converged_count + 1; 
    end 
    
    
    fprintf('Iteration %i: theta = %i \n', iter,theta(1,iter))
end
%%

subplot(2,2,1); 
plot(1:1:length(J),J)

subplot(2,2,2); 
plot(1:1:length(theta),theta)
hold on
plot(1:1:length(theta),K_LQR*ones(1,length(theta)))


subplot(2,2,3); 
plot(1:1:length(grad_J),grad_J)


[~,x_fd] = trajectory_cost(theta(end));

subplot(2,2,4); 
plot(1:1:length(x_lqr),x_lqr)
hold on
plot(1:1:length(x_fd),x_fd)












