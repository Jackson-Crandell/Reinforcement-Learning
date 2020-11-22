function cost = expected_cost(theta)
    global rollouts; 
    
    cost = 0;
    for m = 1:rollouts
        cost = cost + trajectory_cost(theta);
    end
    cost = cost/rollouts; 

end