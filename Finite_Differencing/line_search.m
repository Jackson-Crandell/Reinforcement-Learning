
function alpha = line_search(theta,grad,alpha,c,p)

    debug = false; 

    k = 0; 
    if debug 
        fprintf('Performing line  search..\n') 
    end 
    
    while phi(theta,grad,alpha) <= h(theta,grad,alpha,c)
        k = k + 1; 
        %fprintf('Line search iteration %i  \n', k)
        alpha = p*alpha;
  
    end
    
    if debug
        fprintf('Line search complete in  %i iterations  \n', k)
    end 
    
end  

function out = phi(theta,grad,alpha)

    theta = theta + grad; 
    out = compute_trajectory_cost(theta);

end 

function out = h(theta,grad,alpha,c)

    f = compute_trajectory_cost(theta); 
    out = f + alpha*c*grad*(grad);
    
end 

