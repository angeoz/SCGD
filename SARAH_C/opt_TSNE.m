function [resu_obj, resu_cal, resu_norm] = opt_TSNE(data, config)

w = zeros(size(data.P, 1), config.m) + [1:size(data.P,1)]'/size(data.P,1);
w_t = w;
w_fix = w;
x = 0;


resu_obj = zeros(1,config.max_epochs);
grad_cal = 0;
resu_cal = zeros(1, config.max_epochs);
resu_norm = zeros(1, config.max_epochs);
%% initialize y
[g, G, F] = GD(data, w, config.outer_bs);
y = g;
count = 0;
for epoch = 1:config.max_epochs
    tic;
    %Outer loop update w;
    if (config.opt == 1)||(config.opt == 2)
        [g, G, F] = GD(data, w, config.outer_bs);
    elseif (config.opt == 0)
        [g, G, F, y] = SCGD(data, w, y, config.outer_bs, config.beta);
    elseif (config.opt == 3)
        [g, G, F, y] = ASCPG(data, w, w_fix, y, config.outer_bs, config.beta);
    end
    w_fix = w;
    w = w - config.lr * F;
    x = x + w;
    count = count + 1;
    if config.l1 ~= 0
		w = sign(w).* max(0, abs(w)-config.l1);
    end
    g_fix = g; G_fix = G; F_fix = F;
    norm_F = norm(F);
    grad_cal = grad_cal + config.outer_bs * 2 + 1;
    xresu = x/count;
    if config.mm == 1
        [obj, l2] = compute_tsne(data, xresu, config);
    else
        [obj, l2] = compute_tsne(data, w, config);
    end
        
    resu_obj(epoch) = obj;
    resu_norm(epoch) = norm_F;
    resu_cal(epoch) = grad_cal; 
    for iter = 1:config.max_iters
        if config.opt == 2
            %opt == 2 indicates SARAH_C algorithm
            %Inner loop update w;
            [g, G, F] = SARAH(data, w, w_t, g, G, F, config.inner_bs);
            grad_cal = grad_cal + config.inner_bs * 2 + 1; 
            w_t = w;
            w = w - config.lr * F;
            x = x + w;
            count = count + 1;
        elseif config.opt == 1
            [g, G, F] = SARAH(data, w, w_fix, g_fix, G_fix, F_fix, config.inner_bs);
            grad_cal = grad_cal + config.inner_bs * 2 + 1;
            w = w - config.lr * F;
            x = x + w;
            count = count + 1;
        else
            continue;
        end
        if config.l1 ~= 0
            w = sign(w).* max(0, abs(w)-config.l1);
        end
    end
end
end
    
function [g, G, F] = GD(data, w, batch_size)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    DI = Dist(w, 0);
    g = sum(DI(indexes, :), 1) * n/batch_size - 1;
%% compute G
    G = zeros(d, n, n);
    for i =1:n
        mat = -2 * (w(i, :) - w)'.*DI(i, :)*n/batch_size;
        mat(:, i) = -2 * (w(i, :) - w(indexes, :))' * DI(indexes, i)*n/batch_size;
        G(:, :, i) = mat;
    end
%% compute F
    sample_F = indexes(1);
    F_dev_ = zeros(n, d);
    for i=1:n
        F_dev_(i, :) = 4 * n * sum((w(i, :) - w(sample_F, :)).* dataP(sample_F, i), 1)*n;
    end
    F_dev = (sum(dataP(sample_F, :), 1) * n .* (1./g))';
%% compute gradient
    grad = zeros(n, d);
    for i=1:n
        grad(i, :) = G(:, :, i) * F_dev;
    end
    F = grad + F_dev_;
end

function [g, G, F, y] = SCGD(data, w, y, batch_size, beta)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    DI = Dist(w, 0);
    g = sum(DI(indexes, :), 1) * n/batch_size - 1;
%% compute auxillary
    y = (1-beta) * y + beta * g;
%% compute G
    G = zeros(d, n, n);
    for i =1:n
        mat = -2 * (w(i, :) - w)'.*DI(i, :)*n/batch_size;
        mat(:, i) = -2 * (w(i, :) - w(indexes, :))' * DI(indexes, i)*n/batch_size;
        G(:, :, i) = mat;
    end
%% compute F
    sample_F = indexes(1);
    F_dev_ = zeros(n, d);
    for i=1:n
        F_dev_(i, :) = 4 * n * sum((w(i, :) - w(sample_F, :)).* dataP(sample_F, i), 1)*n;
    end
    F_dev = (sum(dataP(sample_F, :), 1) * n .* (1./y))';
%% compute gradient
    grad = zeros(n, d);
    for i=1:n
        grad(i, :) = G(:, :, i) * F_dev;
    end
    F = grad + F_dev_;
end

function [g, G, F, y] = ASCPG(data, w, w_t, y, batch_size, beta)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% update auxillary    
    z = (1-1/beta) * w_t + (1/beta)*w;
%% compute g
    DI = Dist(w, 0);
    g = sum(DI(indexes, :), 1) * n/batch_size - 1;
    
%% update auxillary
    y = (1-beta) * y + beta * g;
%% compute G
    G = zeros(d, n, n);
    for i =1:n
        mat = -2 * (w(i, :) - w)'.*DI(i, :)*n/batch_size;
        mat(:, i) = -2 * (w(i, :) - w(indexes, :))' * DI(indexes, i)*n/batch_size;
        G(:, :, i) = mat;
    end
%% compute F
    sample_F = indexes(1);
    F_dev_ = zeros(n, d);
    for i=1:n
        F_dev_(i, :) = 4 * n * sum((w(i, :) - w(sample_F, :)).* dataP(sample_F, i), 1)*n;
    end
    F_dev = (sum(dataP(sample_F, :), 1) * n .* (1./y))';
%% compute gradient
    grad = zeros(n, d);
    for i=1:n
        grad(i, :) = G(:, :, i) * F_dev;
    end
    F = grad + F_dev_;
end

function [g, G, F] = SARAH(data, w, w_t, g, G, F, batch_size)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    DI = Dist(w, 0);
    g_mat = sum(DI(indexes, :), 1) * n/batch_size - 1;
    
    DI = Dist(w_t, 0);
    g_mat_t = sum(DI(indexes, :), 1) * n/batch_size - 1;
    g_t = g;
    g = g_mat - g_mat_t + g;
%% compute G
    G_t = G;
    G_ = zeros(d, n, n);
    for i =1:n
        mat = -2 * (w(i, :) - w)'.*DI(i, :)*n/batch_size;
        mat(:, i) = -2 * (w(i, :) - w(indexes, :))' * DI(indexes, i)*n/batch_size;
        G_(:, :, i) = mat;
    end
    
    G_i_t = zeros(d, n, n);
    for i =1:n
        mat_t = -2 * (w_t(i, :) - w_t)'.*DI(i, :)*n/batch_size;
        mat_t(:, i) = -2 * (w_t(i, :) - w_t(indexes, :))' * DI(indexes, i)*n/batch_size;
        G_i_t(:, :, i) = mat_t;
    end
    
    G = G_ - G_i_t + G;
%% compute F
    indexes = randperm(n);
    sample_F = indexes(1);
    F_dev_ = zeros(n, d);
    for i=1:n
        F_dev_(i, :) = 4 * n * sum((w(i, :) - w(sample_F, :)).* dataP(sample_F, i), 1)*n;
    end
    F_dev = (sum(dataP(sample_F, :), 1) * n .* (1./g))';
    
    
    F_dev_t = zeros(n, d);
    for i=1:n
        F_dev_t(i, :) = 4 * n * sum((w_t(i, :) - w_t(sample_F, :)).* dataP(sample_F, i), 1)*n;
    end
    F_devt = (sum(dataP(sample_F, :), 1) * n .* (1./g_t))';
    
    
%% compute gradient
    grad = zeros(n, d);
    for i=1:n
        grad(i, :) = G(:, :, i) * F_dev;
    end
    
    
    grad_t = zeros(n, d);
    for i=1:n
        grad_t(i, :) = G_t(:, :, i) * F_devt;
    end
    F = F + grad - grad_t + F_dev_ - F_dev_t;
end

%%% The only difference between SARAH and SVRG is that in SVRG, w_t, g_t,
%%% G_t, F_t do not update in each iteration. So we don't have to write
%%% another SVRG function

function [dist_matrix] = Dist(data, sig)
    if sig == 0
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2);
    else
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2/(2 * sig^2));
    end
end

    