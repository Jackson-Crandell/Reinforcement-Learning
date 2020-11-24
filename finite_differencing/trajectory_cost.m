function [cost,x] = trajectory_cost(theta)
    
    global A; 
    global B; 
    global Q; 
    global R; 
    
    global Horizon; 

    global x0; 
    x = x0;
   
    target = 0; 
    
    for k = 1:Horizon
        
        u(1,k) = -theta*x(1,k);
        x(1,k+1) = A*x(1,k) + B*u(1,k);

        %Compute the running cost 
        r(1,k) = (x(1,k) - target)'*Q*(x(1,k) - target) + u(1,k)'*R*u(1,k);       
    end
    
    cost = -sum(r(1,:));
end 