
function [F, grad_l2] = compute_tsne(data, w, config)
    dataP = data.P; 
    DI = Dist(w, 0);
    DI2 = squareform(pdist(w, 'euclidean')).^2;
    %% compute G 
    G = sum(DI, 1)-1;
    %% compute F
    F = sum(sum(dataP.*DI2)) + sum(sum(dataP.*log(G)));
    
    grad = GD(data, w);
    grad = grad(:);
    grad_l2 = mean(grad.^2);
end



function [out] = GD(data, w)
    n = size(data.P,1);
    d = size(w, 2);
    dataP = data.P;
    %% compute G 
    DI = Dist(w, 0);
    G = sum(DI, 1)-1;
    %% compute G'
    G_dev = zeros(d, n, n);
    for i = 1:n
        %mat = zeros(d, n);
        mat = -2 * (w(i,:)- w)'.* DI(i,:);
        mat(:, i) = -2 * (w(i,:) - w)' * DI(:,i);
        G_dev(:, :, i) = mat;
    end
    %% Compute F'
    F1_dev = zeros(n, d);
    for i=1:n
        F1_dev(i,:) = 4 * n * sum((w(i,:) - w).* dataP(:,i), 1);
    end
    F_dev = (sum(dataP, 1) .* (1./G))';
    %% update value: G' * F'
    grad = zeros(n, d);
    for i=1:n
        grad(i,:) = G_dev(:,:,i) * F_dev;
    end
    grad = grad + F1_dev;
    out = grad;
end
%Get the distance matrix
function [dist_matrix] = Dist(data, sig)
    if sig == 0
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2);        
    else
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2./(2 * sig^2));
    end
end
