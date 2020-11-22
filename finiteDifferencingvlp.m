%% Finite Differencing
% Dynamics
A = [.4];
B = [.9];
Q = [0.01];
R = [0.001];
dt = 0.001;

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Horizon = 300; %N
rollouts = 300; %M

u = zeros(1,Horizon-1);
x = zeros(1,Horizon);
r = zeros(1,Horizon);

delta_theta = zeros(rollouts,1);
delta_J = zeros(rollouts,1);

sigma = 0.01;    % noise level 
alpha = 0.1;    % learning rate 
iter = 0;       % number of iterations

theta = 0;      % initialize theta
x(1,1) = 0.5;   % initialize initial state

u0 = theta*x(1,1);  
J = x(1,1)'*Q*x(1,1) + u0'*R*u0;   % Set the initial cost 


while iter < 10 % Change to while loop and add stopping condition
    
    iter = iter + 1; 
    expected_reward = 0; 
    
    % Every itetation compute m rollouts
    for m = 1:rollouts
        
        % Perturb the parameter in the policy 
        delta_theta(m,1) = (2*rand - 1)*sigma;  % randomn number bw [-1,1]*sigma
        theta = theta + delta_theta(m,1);  
        
        % Compute the trajectory using perturbed policy
        for k = 1:Horizon
            u(1,k) = theta*x(1,k);
            x(1,k+1) = x(1,k) + A*x(1,k)*dt + B*u(1,k)*dt; 
            
            %Compute the running cost 
            r(1,k) = x(1,k)'*Q*x(1,k) + u(1,k)'*R*u(1,k);       
        end
       
        % total reward of the current rollout
        new_J = sum(r(1,:)); 
        
        % store the difference in the reward between current rollout and 
        % reward at previous iteration
        delta_J(m,1) = new_J - J(iter);
        
        % Add to the expected reward
        expected_reward = expected_reward + sum(r(1,:));
    end
    
    % Compute the expected reward over all rollouts 
    J(1,iter+1) = expected_reward/rollouts;
    
    % Compute the gradient using the Finite Difference Method
    grad_J = (inv(delta_theta' * delta_theta)) * delta_theta' * delta_J
    
    % Update the parameter in the policy 
    theta = theta + alpha*grad_J
end




plot(1:1:length(J),J)




