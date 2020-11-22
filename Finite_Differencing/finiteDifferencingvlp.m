%% Finite Differencing
% Dynamics
global A; 
global B; 
global Q; 
global R; 

global Horizon; 
global dt; 
global rollouts; 

A = [.4];
B = [.9];
Q = [0.01];
R = [0.001];
dt = 0.001;

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Horizon = 300;        %N
rollouts = 100; %M

u = zeros(1,Horizon-1);
r = zeros(1,Horizon);

delta_theta = zeros(rollouts,1);
delta_J = zeros(rollouts,1);

sigma = 0.01;    % noise level 
iter = 0;       % number of iterations

theta = 0.0;      % initialize theta

% Line search parameters
alpha = 0.01;    % learning rate 
c = 0.3 
p = 0.9


global x0; 
x0 = 0.5;   % initialize initial state
J = compute_expected_reward(theta);   % Set the initial cost 

theta_prev = -10; 
eps = 0.0001
%while theta - theta_prev > eps || iter < 100 % Change to while loop and add stopping condition
 
while iter < 1000
    iter = iter + 1; 
    expected_reward = 0; 
    
    % Every itetation compute m rollouts
    for m = 1:rollouts
        
        % Perturb the parameter in the policy 
        delta_theta(m,1) = (2*randn - 1)*sigma;  % randomn number bw [-1,1]*sigma
        new_theta = theta + delta_theta(m,1);  
        
        % total reward of the current rollout
        new_J = compute_trajectory_cost(new_theta);
        
        % store the difference in the reward between current rollout and 
        % reward at previous iteration
        delta_J(m,1) = new_J - J(iter);
        
        % Add to the accumulated reward over every rollout
        expected_reward = expected_reward + new_J;
    end
    
    % Compute the expected reward over all rollouts 
    %J(1,iter+1) = expected_reward/rollouts;
    
     
    % Compute the gradient using the Finite Difference Method
    grad_J = (inv(delta_theta' * delta_theta)) * delta_theta' * delta_J
    
    % Update the parameter in the policy 
    % backtracking line search
    %alpha = line_search(theta,grad_J,alpha,c,p);
    
    theta_prev = theta
    
    theta = theta + alpha*grad_J
    
    J(1,iter+1) =compute_expected_reward(theta);
    
    %J(1,iter+1) = compute_trajectory_cost(theta);
  
    plot(1:1:length(J),J)
    
end




function reward = compute_expected_reward(theta)
    global rollouts; 
    
    reward = 0;
    for m = 1:rollouts
        reward = reward + compute_trajectory_cost(theta);
    end
    reward = reward/rollouts; 

end












