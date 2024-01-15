clear all
close all
YanFun=Yan_functions;
seed=1234;

%% load dataset
% Urban DD
% [Xdata,x_lin,pdf_data]=YanFun.load_UrbanDD();
% 
% % CORS
% % [Xdata,x_lin,pdf_data]=YanFun.load_RefDD();
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% gmm_dist=YanFun.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFun.inflate_GMM(gmm_dist,1,1.2);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';

% GNSS
% [Xdata,x_lin,pdf_data,cdf_data]=YanFun.load_GNSS();
% gmm_dist=[];

% GMM
% [Xdata,x_lin,pdf_data,cdf_data,gmm_dist]=YanFun.load_GMM(seed);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';

% NIG
% [Xdata,x_lin,pdf_data,cdf_data]=YanFun.load_NIG();
% gmm_dist=[];

%% overbound
% % Two-step Gaussian overbound (zero-mean)
% [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFun.two_step_bound_zero(Xdata,x_lin);
% 
% % Two-step Gaussian overbound (same bias)
% [params_tsgo,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFun.two_step_bound(Xdata,x_lin);
% 
% % Gaussian-Pareto overbound (zero-mean)
% [params_gpo,pdf_gpo,cdf_gpo]=YanFun.Gaussian_Pareto_bound(Xdata,x_lin);
% 
% % Total Gaussian overbound
% [mean_tgo, std_tgo, pdf_tgo, cdf_tgo]=YanFun.total_Gaussian_bound(Xdata,x_lin,gmm_dist);

% % Principal Gaussian overbound (zero-mean)
% [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,0.7);

% % Stable(SaS) overbound (zero-mean)
% [alpha_saso, gama_saso,pdf_saso,cdf_saso]=YanFun.stable_bound(Xdata,x_lin);

%% show pdf
% figure
% subplot(1,2,1)
% % plot(x_lin,pdf_data,'k','LineWidth',2);
% histogram(Xdata,'normalization','pdf')
% hold on
% plot(x_lin,pdf_emp,'k--','LineWidth',2);
% plot(x_lin,pdf_tsgo,'r','LineWidth',2);
% plot(x_lin,pdf_gpo,'b','LineWidth',2);
% plot(x_lin,pdf_tgo,'c','LineWidth',2);
% plot(x_lin,pdf_pgo,'g','LineWidth',2);
% plot(x_lin,pdf_saso,'g','LineWidth',2);
% xlabel('Error','FontSize',12);
% ylabel('PDF','FontSize',12);
% A = legend('sample dist.','emp dist.','two-step','Gaussian Pareto','total','Principal Gaussian','stable bound');
% set(A,'FontSize',12)

%% show cdf
% subplot(1,2,2)
% plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
% hold on
% plot(x_lin,cdf_emp,'k--','LineWidth',2);
% plot(x_lin,cdf_tsgo,'r','LineWidth',2);
% plot(x_lin,cdf_gpo,'b','LineWidth',2);
% plot(x_lin,cdf_tgo,'c','LineWidth',2);
% plot(x_lin,cdf_pgo,'g','LineWidth',2);
% % plot(x_lin,cdf_saso,'g','LineWidth',2);
% xlabel('Error','FontSize',12);
% ylabel('CDF','FontSize',12);
% A = legend('sample dist.','emp dist.','two-step','Gaussian Pareto','total','Principal Gaussian','stable bound');
% set(A,'FontSize',12)

%% show log scale CDF
% figure;
% semilogy(x_lin_ecdf,ecdf_data,'ok');
% hold on
% semilogy(x_lin,cdf_emp,'k--','LineWidth',2);
% semilogy(x_lin,cdf_tsgo,'r','LineWidth',2);
% semilogy(x_lin,cdf_gpo,'b','LineWidth',2);
% semilogy(x_lin,cdf_tgo,'c','LineWidth',2);
% semilogy(x_lin,cdf_pgo,'g','LineWidth',2);
% plot(x_lin,cdf_saso,'g','LineWidth',2);
% xlabel('Error','FontSize',12);
% ylabel('CDF','FontSize',12);
% A = legend('sample dist.','emp dist.','two-step','Gaussian Pareto','total','Principal Gaussian','stable bound');
% set(A,'FontSize',12)

%% show QQ plot
% figure;
% % define quantile range
% q_lin=linspace(0.001, 1-0.001, length(Xdata));
% cdf_norm=normcdf(x_lin,0,1);
% ref_q=interp1(cdf_norm, x_lin, q_lin, 'linear', 'extrap');
% data_source = interp1(ecdf_data, x_lin_ecdf, q_lin, 'linear', 'extrap');
% data_emp = interp1(cdf_emp, x_lin, q_lin, 'linear', 'extrap');
% data_tsgo = interp1(cdf_tsgo, x_lin, q_lin, 'linear', 'extrap');
% data_gpo = interp1(cdf_gpo, x_lin, q_lin, 'linear', 'extrap');
% data_tgo = interp1(cdf_tgo, x_lin, q_lin, 'linear', 'extrap');
% data_pgo = interp1(cdf_pgo, x_lin, q_lin, 'linear', 'extrap');
% scatter(data_source,ref_q);
% hold on
% plot(data_emp,ref_q,'k--','LineWidth',2);
% plot(data_tsgo,ref_q,'r','LineWidth',2);
% plot(data_gpo,ref_q,'b','LineWidth',2);
% plot(data_tgo,ref_q,'c','LineWidth',2);
% plot(data_pgo,ref_q,'g','LineWidth',2);
% xlabel('Quantiles of error distribution','FontSize',12);
% ylabel('Standard normal quantile','FontSize',12);
% A = legend('sample dist.','emp dist.','two-step','Gaussian Pareto','total','Principal Gaussian');
% set(A,'FontSize',12)

%% computation time of convolution: fft v.s. direct
% num_conv_list=[1,5,10,20,50,100];
% for i=1:length(num_conv_list)
%     num_conv=num_conv_list(i);
%     ts_fft=0;
%     for j=1:10
%         [pdf_convfft,ts_fft_all]=YanFun.distSelfConv(x_lin,pdf_pgo,num_conv,"fft");
%         ts_fft=ts_fft+ts_fft_all;
%     end
%     fprintf('%i x fft ifft convolution takes: %f s\n',num_conv,ts_fft/10);
%     
%     ts=0;
%     for j=1:10
%         [pdf_conv,ts_all]=YanFun.distSelfConv(x_lin,pdf_pgo,num_conv,"direct");
%         ts=ts+ts_all;
%     end
%     fprintf('%i x direct convolution taks: %f s\n',num_conv,ts/10);
% end

%% Convolution visualization: verify overbound preservation
% YanFun.compareConvOverbound(x_lin,pdf_pgo,pdf_emp,2)
