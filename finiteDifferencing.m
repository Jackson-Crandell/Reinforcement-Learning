%% Finite Differencing
% Dynamics
A = [.01];
B = [.01];
Q = [.01];
R = [.01];
dt = 0.001;

% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R)

Horizon = 300; %N
rollouts = 100; %M
u = zeros(1,Horizon-1);
x = zeros(1,Horizon);
r = zeros(1,Horizon);
J = zeros(1,Horizon);
delta_theta = zeros(1,rollouts);
delta_J = zeros(1,rollouts);
theta = randn; %Randomly initialize theta
expected_reward = zeros(1,rollouts)

while true % Change to while loop and add stopping condition
    for m = 1:rollouts
        delta_theta(:,m) = randn;
        theta = theta + delta_theta(:,m);  
        for k = 1:Horizon
            u(:,k) = u(:,k) * theta;
            x(:,k+1) = x(:,k) + A * dt + B * u(:,k) * dt; %Trajectory
            r(:,k) = x(:,k)'*Q*x(:,k) + u(:,k)'*R*u(:,k); %Cost function
        end
        new_J = r';
        delta_J(:,m) = new_J - J;
        J = new_J;
        % Add stopping condition
        expected_reward(:,m) = 1/iter * sum(r);
    end
    grad_J = (inv(delta_theta(:,m)' * delta_theta(:,m))) * delta_theta' * delta_J;
    theta = theta + grad_J
end
