function cost = compute_trajectory_cost(theta)
    
    global A; 
    global B; 
    global Q; 
    global R; 
    global Horizon; 
    global dt; 
    global x0; 
    
    x = x0; 
    for k = 1:Horizon
        u(1,k) = theta*x(1,k);
        x(1,k+1) = x(1,k) + A*x(1,k)*dt + B*u(1,k)*dt ;

        %Compute the running cost 
        r(1,k) = x(1,k)'*Q*x(1,k) + u(1,k)'*R*u(1,k);       
    end
    
    cost = sum(r(1,:));
end 