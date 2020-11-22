%% Finite Differencing
% Dynamics
A = [.1];
B = [.5];
Q = [0.1];
R = [0.01];
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

sigma = 0.05;
alpha = 0.1;
iter = 0;


theta = randn; 
x(1,1) = 0.5; 


u0 = theta*x(1,1);
J = -x(1,1)'*Q*x(1,1) - u0'*R*u0;       %Cost function



while iter < 10 % Change to while loop and add stopping condition
    
    iter = iter + 1; 
    expected_reward = 0; 
    
    for m = 1:rollouts
        
        delta_theta(m,1) = (2*rand - 1)*sigma;
        theta = theta + delta_theta(m,1);  
        
        for k = 1:Horizon
            u(1,k) = theta*x(1,k);
            x(1,k+1) = x(1,k) + A*x(1,k)*dt + B*u(1,k)*dt + (2*rand - 1)*sigma; %Trajectory
            r(1,k) = x(1,k)'*Q*x(1,k) + u(1,k)'*R*u(1,k);       %Cost function
        end
       
        new_J = sum(r(1,:));
        delta_J(m,1) = new_J - J(iter);
        
        expected_reward = expected_reward + sum(r(1,:));
    end
    
    J(1,iter+1) = expected_reward/rollouts;
    
    grad_J = (inv(delta_theta' * delta_theta)) * delta_theta' * delta_J
    theta = theta - alpha*grad_J
end




plot(1:1:length(J),J)




