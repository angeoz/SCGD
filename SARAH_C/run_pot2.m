rng(1, 'twister');

config.l1 = 5e-5;
config.l1 = 0;
%config.kappa = 20;

%mu = ones(1, 200)*4;
%A = randn(200, 200);
%[u, s, v] = svd(A);
%s = eye(200);
%s(200, 200) = config.kappa;
%sigma = u*s*u';
%data = mvnrnd(mu, sigma, 2000);
%save('data_cov_4.mat', 'data');
%Probname = {'Asia_Pacific_ex_Japan_OP', 'Europe_OP', 'Global_ex_US_OP', 'Global_OP', 'Japan_OP', 'North_America_OP'};
%Probname = {'Asia_Pacific_ex_Japan_ME', 'Europe_ME', 'Global_ex_US_ME', 'Global_ME', 'Japan_ME', 'North_America_ME'}; 
%Probname = {'Japan_OP'};
%Probname = {'Global_ex_US_OP', 'Japan_OP', 'Global_OP', 'Asia_Pacific_ex_Japan_OP', 'Europe_OP', 'North_America_OP'};
%Titlename = {'Global ex US OP', 'Japan OP', 'Global OP', 'Asia Pacific ex Japan OP', 'Europe OP', 'North America OP'};
Probname = {'data_cov_4', 'data_cov_20'};
Titlename = {'data cov 4', 'data cov 20'};
%Probname = {'ME'};
%Probname = {'Global_ex_US_OP'};
%Probname = {'Europe_OP'};
%Probname = {'North_America_OP'};
lrlist = [[1e-3, 5e-3, 5e-3]; [1e-3, 2e-3, 5e-4]; [1e-3, 2e-3, 5e-4];[1e-3, 5e-3, 5e-4];[1e-4, 1e-3, 5e-4];[1e-3, 2e-3, 5e-4]];
lrlist = [[1e-3, 5e-3, 5e-3]];%Asia_Pacific
%lrlist = [[5e-5, 2e-4, 2e-4]];
lrlist = [[1e-4, 1e-2, 1.1e-2]];%Global_ex_US_OP
%lrlist = [[1e-4, 1e-3, 1e-3]];%Japan_OP
lrlist = [2e-3, 1e-3, 1e-3, 2e-3, 1e-3, 1e-3];
lrlist = [2e-5, 2e-5];
nprob = length(Probname);
Problist = [1:nprob];
figure;
config.m = 1;
config.m = 0;
for di = 1:length(Problist) 
    
    %% load data
    probID = Problist(di);
    name = Probname{probID};
    load(strcat('./data/', Probname{Problist(di)},'.mat'));
    %load data_cov_2;
    [n, d] = size(data);
    config.lr = lrlist(di);

rng(1);
minval = compute_min_val(data, config);

config.gamma = 0.95;
config.max_iters = 20; 
config.max_epochs = 500;
config.outer_bs = 2000;
config.inner_bs = 5;

config.beta = 0.9;
config.opt = 1;
[svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
grad_svrg = grad_svrg/n;
config.opt = 2;
config.dec = 1;
[spider, grad_spider, norm_spider] = opt_VR(data, config);
grad_spider = grad_spider/n;
config.dec = 0;
config.opt = 0;
[scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
grad_scgd = grad_scgd/n;
config.opt = 3;
[ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
grad_ascpg = grad_ascpg/n;
%config.opt = 3;
%config.lambda = 1;
%config.lr = 3e-5;
%config.max_iters = 1;
%[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
%config.lr = 5e-4;
%[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
%subplot(2, 3, di);

subplot(2, 2, (di-1)*2+1);
semilogy(grad_svrg, smooth(svrg-minval, 10), '-o', 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
hold on;
semilogy(grad_spider, smooth(spider-minval, 10),'-*', 'Color',[0.9290 0.6940 0.1250], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
semilogy(grad_scgd, smooth(scgd-minval, 10), '--','Color', [0.6350 0.0780 0.1840],  'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
semilogy(grad_ascpg, smooth(ascpg-minval, 10), ':', 'Color', [0.3010 0.7450 0.9330], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
xlabel('Grads Calculation/n');
ylabel('Objective Value Gap');
title(Titlename(di))
hold off;


subplot(2, 2, (di-1)*2+2);
semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
xlabel('Grads Calculation');
ylabel('Gradient Norm');
title(Titlename(di));

% 
% 
% 
% 
% 
% config.lr = 2e-4;
% config.beta = 0.9;
% config.opt = 1;
% [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
% config.opt = 2;
% config.dec = 1;
% [spider, grad_spider, norm_spider] = opt_VR(data, config);
% config.dec = 0;
% config.opt = 0;
% config.lr = 2e-4;
% [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
% config.opt = 3;
% config.lr = 2e-4;
% [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
% %config.opt = 3;
% %config.lambda = 1;
% %config.lr = 3e-5;
% %config.max_iters = 1;
% %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
% %config.lr = 5e-4;
% %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
% figure;
% subplot(1, 2, 1);
% semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Objective Value Gap');
% title('Objective Value Gap vs. Grads Calculation')
% 
% subplot(1, 2, 2);
% semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Gradient Norm');
% title('Gradient Norm vs. Grads Calculation')
% 
% 
% 
% 
% 
% 
% 
% config.lr = 3e-4;
% config.beta = 0.9;
% config.opt = 1;
% [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
% config.opt = 2;
% config.dec = 1;
% [spider, grad_spider, norm_spider] = opt_VR(data, config);
% config.dec = 0;
% config.opt = 0;
% config.lr = 3e-4;
% [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
% config.opt = 3;
% config.lr = 3e-4;
% [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
% %config.opt = 3;
% %config.lambda = 1;
% %config.lr = 3e-5;
% %config.max_iters = 1;
% %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
% %config.lr = 5e-4;
% %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
% figure;
% subplot(1, 2, 1);
% semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Objective Value Gap');
% title('Objective Value Gap vs. Grads Calculation')
% 
% subplot(1, 2, 2);
% semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Gradient Norm');
% title('Gradient Norm vs. Grads Calculation')
% 
% 
% 
% 
% 
% 
% 
% 
% config.lr = 5e-4;
% config.beta = 0.9;
% config.opt = 1;
% [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
% config.opt = 2;
% config.dec = 1;
% [spider, grad_spider, norm_spider] = opt_VR(data, config);
% config.dec = 0;
% config.opt = 0;
% config.lr = 5e-4;
% [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
% config.opt = 3;
% config.lr = 5e-4;
% [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
% %config.opt = 3;
% %config.lambda = 1;
% %config.lr = 3e-5;
% %config.max_iters = 1;
% %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
% %config.lr = 5e-4;
% %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
% figure;
% subplot(1, 2, 1);
% semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Objective Value Gap');
% title('Objective Value Gap vs. Grads Calculation')
% 
% subplot(1, 2, 2);
% semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Gradient Norm');
% title('Gradient Norm vs. Grads Calculation')
% 
% 
% 
% 
% 
% 
% 
% config.lr = 8e-4;
% config.beta = 0.9;
% config.opt = 1;
% [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
% config.opt = 2;
% config.dec = 1;
% [spider, grad_spider, norm_spider] = opt_VR(data, config);
% config.dec = 0;
% config.opt = 0;
% config.lr = 8e-4;
% [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
% config.opt = 3;
% config.lr = 8e-4;
% [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
% %config.opt = 3;
% %config.lambda = 1;
% %config.lr = 3e-5;
% %config.max_iters = 1;
% %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
% %config.lr = 5e-4;
% %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
% figure;
% subplot(1, 2, 1);
% semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Objective Value Gap');
% title('Objective Value Gap vs. Grads Calculation')
% 
% subplot(1, 2, 2);
% semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
% legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
% xlabel('Grads Calculation');
% ylabel('Gradient Norm');
% title('Gradient Norm vs. Grads Calculation')
% %plot(grad_svrg, svrg, 'b', grad_spider,spider, 'r', grad_scgd, scgd, 'g');
% %legend('Svrg', 'Spider-A', 'SCGD');
% %xlabel('Num Iter');
% %ylabel('Objective Value');
% %title('Best tuned lrscgd=1e-3, lrsvrg=5e-3, lrspider=1e-2, iters=50, A=10, B=10, eps=1e-2, *10*0.8^{epoch}')

end