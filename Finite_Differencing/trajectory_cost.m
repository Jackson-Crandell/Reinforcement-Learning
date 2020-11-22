function [cost,x] = trajectory_cost(theta)
    
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
        
%         u(1,k) = -theta*x(1,k);
%         x(1,k+1) = x(1,k) + A*x(1,k)*dt + B*u(1,k)*dt ;

        u(1,k) = -theta*x(1,k);
        x(1,k+1) = A*x(1,k) + B*u(1,k); % + (2*rand - 1)*sigma;

        %Compute the running cost 
        r(1,k) = (x(1,k) - target)'*Q*(x(1,k) - target) + u(1,k)'*R*u(1,k);       
        
    end
    
    % reward = -cost
    cost = -sum(r(1,:));
end 