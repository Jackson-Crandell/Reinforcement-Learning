%% Finite Differencing
% Dynamics
A = [.01];
B = [.01];
Q = [.01];
R = [.01];
Horizon = 300; %N
rollouts = 100; %M
% Optimal Control Gain
[K_LQR,S,E] = dlqr(A,B,Q,R)
u = zeros(1,Horizon-1);
x = zeros(1,Horizon);
r = zeros(1,Horizon);
J = 0;
delta_theta = zeros(1,rollouts);
delta_J = zeros(1,rollouts);
for j = 1:iter
    for m = 1:rollouts
        for k = 1:Horizon
            delta_theta(:,m) = randn;
            x(:,k+1) = x(:,k) + A * dt + B * u(:,k) * dt + randn; %Trajectory
            r(:,k) = x'*Q*x + u'*R*u; %Cost function
            new_J = r(:,k);
            delta_J(:,m) = new_J - J;
            J = new_J;
            grad_J = (inv(delta_theta' * delta_theta)) * delta_theta' * delta_J;
            theta = theta + grad_J;
            u(:,k) = theta * x(:,k); %Add noise? u = Kx
            expected_reward = 1/iter * sum(r(:,:));
        end
    end

end
