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

x0 = 1;   % initialize initial state
Horizon = 300;        %N
rollouts = 100;

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R);

%[~,x_lqr] = trajectory_cost(K_LQR);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(1,Horizon-1);
r = zeros(1,Horizon);

global sigma2;
sigma2 = 1e-4;    % policy noise level 

theta = 0.0;      % initialize theta
alpha = 0.2;    % learning rate 

grad_J = 0; 
eps = 1e-4;

iter = 0;       % number of iterations
converged_count = 0; 

while converged_count < 10
    
    iter = iter + 1;
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample trajectories with current policy
    for m = 1:rollouts
        [running_cost(1,m), epsilon(:,m)] = trajectory_cost(theta(1,iter),1);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the gradient using the REINFORCE Method
    expectation = 0; 
    for j = 1:rollouts
        expectation = expectation + running_cost(1,j)*(sum(epsilon(:,j))); 
    end 
   
    grad_J(1,iter) = expectation/rollouts;
    
    % Update the parameter in the policy 
    %alpha = 1/sqrt(alpha)
    theta(1,iter+1) = theta(1,iter) + alpha*grad_J(1,iter);
   
    grad_converged = abs(grad_J(1,iter));

    if grad_converged < eps
        converged_count = converged_count + 1; 
    end 
    
    [J(1,iter), ~] = trajectory_cost(theta(1,iter),0);
    
    fprintf('Iteration %i: theta = %i , Cost = %i, grad_J = %i \n', iter,theta(1,iter),J(1,iter),grad_J(1,iter)); 
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


%[~,x_fd] = trajectory_cost(theta(end));

% subplot(2,2,4); 
% plot(1:1:length(x_lqr),x_lqr)
% hold on
% plot(1:1:length(x_fd),x_fd)












