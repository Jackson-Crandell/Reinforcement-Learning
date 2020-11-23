function [cost,x,epsilon,b] = trajectory_cost(theta)
    
    global A; 
    global B; 
    global Q; 
    global R; 
    
    global Horizon; 
    global dt; 
    
    global x0; 
    global sigma;
    
    x = x0;
    target = 0; 
    
    for k = 1:Horizon
        
        variance = 1;
        gaussian(k) = sqrt(variance)*randn(N,1) + 1; %Gaussian with mean 1 and variance 
        b(k) = variance;
        epsilon(k) = gaussian(k);
        
        u(1,k) = -theta*x(1,k) + gaussian(k);
        x(1,k+1) = A*x(1,k) + B*u(1,k);

        %Compute the running cost 
        r(1,k) = (x(1,k) - target)'*Q*(x(1,k) - target) + u(1,k)'*R*u(1,k);       
        
    end
    
    % reward = -cost
    cost = -sum(r(1,:));
end 