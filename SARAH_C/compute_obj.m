
function [F, grad_l2] = compute_obj(data, w, config)
    dataF = data.F;
    dataR = data.R;
    dataP = data.P;
    %% compute G 

    G_1 = dataF(:,:) * w';
    G_2 = sum(dataP.* (dataR + config.gamma * (dataF * w')'), 2);

    %% compute F
    F = sum((G_1 - G_2).^2);
    
    grad = GD(data, w, config);
    grad = grad(:);
    grad_l2 = sum(grad.^2);
end



function [out] = GD(data, w, config)
    n = size(data.F, 1);
    d = size(data.F, 2);
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;

    %% compute G
    
    G_1 = dataF(:,:) * w';
    G_2 = sum(dataP.* (dataR + config.gamma * (dataF * w')'), 2);
    G = zeros(1, 2*n);
    G(1:2:end) = G_1;
    G(2:2:end) = G_2;
    

    %% compute G'
    G_dev_1 = dataF';
    G_dev_2 = (dataP * config.gamma * dataF)';
    G_dev = zeros(d, 2*n);
    G_dev(:, 1:2:end) = G_dev_1;
    G_dev(:, 2:2:end) = G_dev_2;

    %% compute F'
    mid = 2 * (G(1:2:end) - G(2:2:end));
    F_dev_ = zeros(2*n, 1);
    F_dev_(1:2:end) = mid;
    F_dev_(2:2:end) = -mid;
    
    F_dev = G_dev * F_dev_;
    

	out = F_dev;%vertical
end

