%% Finite Differencing
% Dynamics
A = [.4];
B = [.9];
Q = [.1];
R = [.001];

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R)

delta_theta = 0:0.01:0.05;
theta = -0.4;
vec_theta = theta + pert;
alpha = 0.01;
Horizon = 500; %N
rollouts = length(pert); %M
iterations = 400;


for i = 1:iterations % Change to while loop and add stopping condition
    u = zeros(rollouts,Horizon-1);
    x = zeros(rollouts,Horizon);
    x(:,1) = 1; %Starting at position 1
    J = zeros(1,rollouts);

    for m = 1:rollouts
        for k = 1:Horizon-1
            u(m,k) = vec_theta(m)*x(m,k);
            x(m,k+1) = A*x(m,k)  + B*u(m,k); %Trajectory
            J(m) = J(m) + x(m,k)'*Q*x(m,k) + u(m,k)'*R*u(m,k); %Cost function
        end
    end
    delta_J = J - J(1);
    grad_J = (inv(delta_theta * delta_theta')) * delta_theta * delta_J';
    theta = theta + -1*alpha*grad_J
end

%%
%% Finite Differencing
% Dynamics
A = [.4];
B = [.9];
Q = [.1];
R = [.001];

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R)

theta = zeros(1,rollouts);
alpha = 0.1;
Horizon = 300; %N
rollouts = 400; %M
iterations = 500;
delta_theta = zeros(1,rollouts);

for i = 1:iterations % Change to while loop and add stopping condition
    u = zeros(rollouts,Horizon-1);
    x = zeros(rollouts,Horizon);
    x(:,1) = 1; %Starting at position 1
    J = zeros(1,rollouts);
    theta = zeros(1,rollouts);

    for m = 1:rollouts
        for k = 1:Horizon-1
            delta_theta(m) = rand;
            theta(m) = theta(m) + delta_theta(m);
            u(m,k) = theta(m)*x(m,k);
            x(m,k+1) = A*x(m,k)  + B*u(m,k); %Trajectory
            J(m) = x(m,k)'*Q*x(m,k) + u(m,k)'*R*u(m,k); %Cost function
        end
    end
    delta_J = J - J(1);
    grad_J = (inv(delta_theta * delta_theta')) * delta_theta * delta_J';
    theta(m) = theta(m) + alpha*grad_J
end


