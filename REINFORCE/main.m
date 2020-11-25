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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(1,Horizon-1);
r = zeros(1,Horizon);

global sigma;
sigma = 0.01;    % policy noise level 

theta = 0.0;      % initialize theta
alpha = 0.3;    % learning rate 

grad_J = 0; 
eps = 1e-8;

[J, ~] = trajectory_cost(theta,0);

iter = 0;       % number of iterations
converged_count = 0; 
cost_change = 0; 

% Run algorithm until the gradient converges or the cost changes direction
while converged_count < 10 && cost_change < 1
    
    iter = iter + 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample trajectories with current policy
    for m = 1:rollouts
        [running_cost(1,m),epsilon(:,m)] = trajectory_cost(theta(1,iter),1);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the gradient using the REINFORCE Method
    expectation = 0; 
    for j = 1:rollouts
        expectation = expectation + running_cost(1,j)*(sum(epsilon(:,j))); 
    end 
   
    grad_J(1,iter) = expectation/rollouts;
    
    % Compute the new theta policy using gradient ascent
    theta(1,iter+1) = theta(1,iter) + alpha*grad_J(1,iter);
   
    grad_converged = norm(grad_J(1,iter));

    if grad_converged < eps
        converged_count = converged_count + 1; 
    end 
    % Compute the cost of the new policy
    [J(1,iter+1), ~] = trajectory_cost(theta(1,iter+1),0);
    
    % Check the cost doesn't change directions
    delta_grad_J = abs(J(1,iter+1)) - abs(J(1,iter)); 
    if delta_grad_J > 0
        cost_change = cost_change + 1 ; 
    end 
    
    fprintf('Iteration %i: theta = %i , Cost = %i, grad_J = %i \n', iter,theta(1,iter),J(1,iter),grad_J(1,iter)); 
    
    
end
%%

subplot(1,3,1); 
plot(1:1:length(J),J,'linewidth',2)
title('$Reward$','Interpreter','latex','fontsize',32);
xlabel('Iteration','fontsize',20);


% LQR outputs a positive K gain. Our controller is u = Kx therefore our 
% theta will be negative. For purposes of comparison we flip theta to be 
% positive and flip the policy gradient to indicate the direction of update
% in this convention
subplot(1,3,2); 
plot(1:1:length(theta),-theta,'linewidth',2)
hold on
plot(1:1:length(theta),K_LQR*ones(1,length(theta)), 'linewidth',4)
title('$K$','Interpreter','latex','fontsize',32);
xlabel('Iteration','fontsize',20);

subplot(1,3,3); 
plot(1:1:length(grad_J),-grad_J,'linewidth',2)
title('$\nabla_{\theta} J$','Interpreter','latex','fontsize',32);
xlabel('Iteration','fontsize',20);









