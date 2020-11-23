%% REINFORCE
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

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R);

%[~,x_lqr] = trajectory_cost(K_LQR);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(1,Horizon-1);
r = zeros(1,Horizon);

%sigma = 0.01;    % noise level 

theta = .25;      % initialize theta
alpha = 0.01;    % learning rate 

grad_J = 0; 
eps = 1e-5;

iter = 0;       % number of iterations

converged_count = 0; 

while converged_count < 10
    
    iter = iter + 1;
     
    expected_reward = 0; 
    
    x = x0;
    target = 0; 
    b = ones(1,Horizon);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    for k = 1:Horizon
        variance = 0.1;
        epsilon(1,k) = randn; %Gaussian with mean 0 and variance 
        b(k) = sqrt(variance);
        u(1,k) = theta(1,iter)*x(1,k) + b(k)*epsilon(1,k);
        x(1,k+1) = A*x(1,k) + B*u(1,k);
        phi(1,k) = x(1,k);
        %Compute the running cost 
        r(1,k) = (x(1,k) - target)'*Q*(x(1,k) - target) + u(1,k)'*R*u(1,k);       
        
    end
    
    % reward = -cost
    reward = -sum(r(1,:));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute the gradient using the REINFORCE Method
    grad_J(1,iter) = (1/Horizon) * reward * sum(phi'*inv(b*b')*b*epsilon')
    
    % Update the parameter in the policy 
    %alpha = 1/sqrt(alpha)
    theta(1,iter+1) = theta(1,iter) + alpha*grad_J(1,iter);
    theta(iter)
   
    grad_converged = abs(grad_J(1,iter));

    if grad_converged < eps
        converged_count = converged_count + 1; 
    end 
    
    
    fprintf('Iteration %i: theta = %i \n', iter,theta(1,iter))
end
%%

subplot(2,2,1); 
plot(1:1:length(iter),iter)

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












