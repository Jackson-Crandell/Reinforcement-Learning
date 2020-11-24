function [cost,x,epsilon] = trajectory_cost(theta,noise)
    
    global A; 
    global B; 
    global Q; 
    global R; 
    
    global Horizon; 
    
    global x0; 
    global sigma;
    
    x = x0;
    target = 0; 
    
    for k = 1:Horizon
        
        if noise == 1
            epsilon(1,k) = sigma*randn; %Gaussian with mean 0 and variance 
        end 
        
        if noise == 0 
            epsilon(1,k) = 0; 
        end 
        
        u(1,k) = theta*x(1,k) + epsilon(1,k)*x(1,k);
        x(1,k+1) = A*x(1,k) + B*u(1,k);

        %Compute the running cost 
        r(1,k) = (x(1,k) - target)'*Q*(x(1,k) - target) + u(1,k)'*R*u(1,k);       
    end

    % reward = -cost
    cost = -sum(r(1,:));
end 