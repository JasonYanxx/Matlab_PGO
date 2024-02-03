% clear all
% close all
seed=1234;
YanFuncLib_Overbound_tmp=YanFuncLib_Overbound;
%% explore T1_trans method (20230701)
% % analysis whether T1_trans can be modelled as an overbound
% [Xdata,x_lin,pdf_data,cdf_data,gmm_dist]=YanFuncLib_Overbound_tmp.load_GMM(seed);
% YanFuncLib_Overbound_tmp.compare_twoside_bound(Xdata,x_lin,gmm_dist)

%% explore Fault detection (20230702)
% % explore how conservatism will affect fault detection performance
% % GMM
% [Xdata,x_lin,pdf_data,cdf_data,gmm_dist]=YanFuncLib_Overbound_tmp.load_GMM(seed);
% % Two-step Gaussian overbound (zero-mean)
% [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_zero(Xdata,x_lin);
% % Total Gaussian overbound
% [mean_tgo, std_tgo, pdf_tgo, cdf_tgo]=YanFuncLib_Overbound_tmp.total_Gaussian_bound(Xdata,x_lin,gmm_dist);
% 
% % FDE params
% alpha=0.05;
% num=6;
% bias=quantile(Xdata,0.001/2);
% % different method
% [FA_Gaussian,MD_Gaussian]=YanFuncLib_Overbound_tmp.FDE_Gaussian(alpha,seed,num,gmm_dist,bias,std_tsgo^2);
% [FA_tGaussian,MD_tGaussian]=YanFuncLib_Overbound_tmp.FDE_Gaussian(alpha,seed,num,gmm_dist,bias,std_tgo^2);
% [FA_Bayes_max,MD_Bayes_max]=YanFuncLib_Overbound_tmp.FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,"max");
% [FA_Bayes_sum,MD_Bayes_sum]=YanFuncLib_Overbound_tmp.FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,"sum");
% % Monte-Carlo simulation
% [FA_arr,MD_arr,var_arr]=YanFuncLib_Overbound_tmp.FDE_mc_compare(alpha,seed,num);

%% generate NIC dist.(20230704)
% global mu;
% global alpha;
% global delta;
% global beta;
%     
% mu=0;
% alpha=0.65;
% delta=0.65;
% beta=0;
% 
% x = linspace(-7, 7, 1000);
% f_sam=nig_pdf(x);
% F_sam=cumtrapz(f_sam)*(x(2)-x(1));
% 
% 
% N = 100000; % number of random numbers to generate
% interval = [-7, 7]; % interval over which pdf is defined
% M = 1; % constant M for acceptance-rejection method
% X = YanFuncLib_Overbound_tmp.customrand(@nig_pdf, interval, N, M)';
% 
% figure
% plot(x,f_sam);
% hold on
% histogram(X,'normalization','pdf')

%% Fourier transform explore(20230706)
% p1=0.6;
% p2=1-p1;
% mu1=0;
% mu2=0;
% delta1=0.1342;
% delta2=0.4399;
% xp=0.4987;
% k=0.0012;
% c=0.2980;
% 
% figure;
% syms x w
% y=(p1/(delta1*sqrt(2*pi))*exp(-0.5*(x/delta1)^2)+c)*(heaviside(x+xp)-heaviside(x-xp));
% y=y+(1+k)*p2/(delta2*sqrt(2*pi))*exp(-0.5*(x/delta2)^2)*heaviside(-x-xp);
% y=y+(1+k)*p2/(delta2*sqrt(2*pi))*exp(-0.5*(x/delta2)^2)*heaviside(x-xp);
% F=fourier(y,x,w);
% ff=ifourier(F*F,w,x);
% fplot(y,[-2,2])
% hold on
% fplot(ff,[-4,4])
% grid
% xlabel("Time")
% ylabel("Amplitude")

%% fft ifft explore(20230707)
% % define error dist. (GMM)
% global p1 p2 mu1 mu2 sigma1 sigma2 xp
% p1=0.9774;
% p2=1-p1;
% mu1=0;
% mu2=0;
% sigma1=0.1342^2;
% sigma2=0.4399^2;
% xp=0.6148;
% gm = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
% 
% % define discrete intervals
% N = 1024;       
% x = linspace(-2, 2, N);  
% dx = x(2) - x(1);
% % discrete pdf cdf of error dist.
% pdf_gm=pdf(gm,x');
% pdf_gm = pdf_gm / sum(pdf_gm*dx); 
% cdf_gm = cdf(gm,x');
% % discrete pdf of overbound
% pdf_pb=YanFuncLib_Overbound_tmp.two_piece_pdf(x,gm,-xp,xp);
% pdf_pb = pdf_pb / sum(pdf_pb*dx);   % normalization
% 
% % fft
% fft_pdf = fft(pdf_pb);
% fft_pdf_norm = fft_pdf / N; % normalization
% freq = linspace(0, 1/2, N/2+1); % range of frequency
% % ifft
% pdf_recon=ifft(fft_pdf);
% 
% % visualization
% subplot(1,3,1) % fft
% stem(freq, abs(fft_pdf_norm(1:N/2+1)));
% title('FFT of Gaussian Distribution');
% xlabel('Frequency');
% ylabel('Amplitude');
% 
% 
% subplot(1,3,2) % ifft
% plot(x,pdf_gm,'k','linewidth',1);
% hold on, grid on
% plot(x, pdf_pb,'b','linewidth',1);
% plot(x, pdf_recon,'r','linewidth',1) % Reconstructed
% xlabel('error'), ylabel('pdf')
% 
% % self-convolution
% % pdf_gm_conv2=conv(pdf_gm,pdf_gm,'same');
% % pdf_gm_conv2=pdf_gm_conv2/sum(pdf_gm_conv2*dx); % normalize PDF
% [pdf_gm_conv2,~]=YanFuncLib_Overbound_tmp.get_conv(x,pdf_gm,pdf_gm);
% plot(x,pdf_gm_conv2,'k--','linewidth',1);
% tic;
% fft_pdf2=fft(pdf_pb,2*N-1);
% pdf_recon2 = ifft(fft_pdf2 .* fft_pdf2);
% fft_conv_t=toc;
% pdf_recon2=pdf_recon2/sum(pdf_recon2*dx); % normalize PDF
% pdf_recon2=pdf_recon2(1,floor(N/2):floor(N/2)+N-1); % cut
% plot(x,abs(pdf_recon2),'b--','linewidth',0.5);
% % tic;
% % pdf_dconv2=conv(pdf_pb,pdf_pb);
% % conv_t=toc;
% % pdf_dconv2=pdf_dconv2/sum(pdf_dconv2*dx); % normalize PDF
% % pdf_dconv2=pdf_dconv2(1,floor(N/2):floor(N/2)+N-1); %cut
% [pdf_dconv2,conv_t]=YanFuncLib_Overbound_tmp.get_conv(x,pdf_pb,pdf_pb);
% plot(x,pdf_dconv2,'r:','linewidth',1);
% A = legend('sample','pb','recon pb','sample conv','fft ifft conv','direct conv');
% set(A,'FontSize',12)
% 
% fprintf('fft ifft convolution taks: %f s\n',fft_conv_t);
% fprintf('direct convolution taks: %f s\n',conv_t);
% 
% subplot(1,3,3) % cdf
% plot(x,cdf_gm,'k','linewidth',1)
% hold on
% cdf_gm_conv2=cumtrapz(pdf_gm_conv2)*(x(2)-x(1));
% plot(x,cdf_gm_conv2,'k--','linewidth',1);
% cdf_recon2=cumtrapz(pdf_recon2)*(x(2)-x(1));
% plot(x,cdf_recon2,'b','linewidth',1);
% cdf_dconv2=cumtrapz(pdf_dconv2)*(x(2)-x(1));
% plot(x,cdf_dconv2,'r','linewidth',1);
% plot(x,sign(cdf_recon2-cdf_gm_conv2'),'m','linewidth',1);
% A = legend('sample','sample conv','fft ifft pb conv','direct pb conv','sign fft ifft pb conv');
% set(A,'FontSize',12)

%% compare fft & conv (20230715)
% x_lin=-5:0.01:5;
% pdf_data=normpdf(x_lin,0,2);
% 
% x=x_lin;
% dt=abs(x_lin(1)-x_lin(2));
% pdf_conv=conv(pdf_data,pdf_data)*dt;
% [pdf_conv_org,x_fftconv,conv_t_org]=YanFuncLib_Overbound_tmp.distConv_org(x,x,pdf_data,pdf_data,"direct");
% [pdf_fftconv_org,x_fftconv_org,fft_conv_t_org]=YanFuncLib_Overbound_tmp.distConv_org(x,x,pdf_data,pdf_data,"fft");
% 
% plot(x_fftconv,pdf_conv_org);
% hold on
% plot(x_fftconv_org,pdf_fftconv_org);


%% Generalized Pareto overbound (20230717)
% % define error dist. (GMM)
% p1=0.9;
% p2=1-p1;
% mu1=0;
% mu2=0;
% sigma1=0.2^2;
% sigma2=1^2;
% gm = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
% Nsamples=10001;
% Xdata=random(gm, Nsamples);
% 
% % right-tail overbound
% global thr_R xi_R scale_R theta_R
% [thr_R,theta_R,xi_R,scale_R]=YanFuncLib_Overbound_tmp.gp_tail_overbound(Xdata);
% % left-tail overbound
% global thr_L xi_L scale_L theta_L
% [thr_L,theta_L,xi_L,scale_L]=YanFuncLib_Overbound_tmp.gp_tail_overbound(-Xdata);
% thr_L=-thr_L;
% % Gaussian core overbound - Two-step method
% Nbins = 100;
% NstepsCdf = 1000;
% epsilon = 0.0025;
% global mean_core std_core
% [mean_core, std_core, epsilon_achieved, intervals]=gaussian_overbound(Xdata, epsilon, Nbins, NstepsCdf);
% mean_core=0;
% 
% % visualization - CDF
% figure(1,2,1)
% lim=max(-min(Xdata),max(Xdata));
% x = linspace(-lim, lim, Nsamples);
% y=cdf(gm, x');
% plot(x,y,'r','LineWidth',2);
% hold on
% y=GaussPareto_cdf(x);
% plot(x,y,'b','LineWidth',2);
% 
% % visualization - PDF
% figure(1,2,2)
% y=pdf(gm, x');
% plot(x,y,'r','LineWidth',2);
% hold on
% y=GaussPareto_cdf(x);
% y_pdf = diff(y) ./ (x(2)-x(1));
% y_pdf(end+1) = y_pdf(end);
% plot(x,y_pdf,'b','LineWidth',2);

%% 2023072 GPS data (from yihan) processing
% % set folder path and file name
% folder = 'Data/mnav_zmp1_jan_20240105'; 
% filePattern = fullfile(folder, '*.csv'); % using wildcard
% csvFiles = dir(filePattern); % 

% % read each csv
% for i = 1:length(csvFiles)
%     filename = fullfile(csvFiles(i).folder, csvFiles(i).name);
%     data{i} = readtable(filename, 'HeaderLines', 1); % jump first line of header 
% end
% 
% % merge
% bigTable = vertcat(data{:});
% % set folder path and file name
% filename = 'merged_Ref_Jan.csv'; % new filename
% folder = 'Data/mnav_zmp1_jan_20240105'; % path for saving new file
% fullPath = fullfile(folder, filename);
% 
% % save to csv
% writetable(bigTable, fullPath, 'Delimiter', ',', 'QuoteStrings', true);
% % add header manually
% 
% load('Data/mnav_zmp1_jan/mergedRefJan.mat');
% % select: ele(25~50)
% filter_ele=(mergedurbandd.U2I_Elevation>=25 && mergedurbandd.U2I_Elevation<=50); 
% Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele);

%% 20230802 Bias in Principal Gaussian overbound
% seed=1234;
% % load GMM
% % [Xemp,x_lin_emp,pdf_emp,cdf_emp,gmm_dist]=YanFuncLib_Overbound_tmp.load_GMM_bias(seed);
% % [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD();
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD();
% % Xdata=Xdata-median(Xdata); % should not do this
% % [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD();
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% counts=length(x_lin);
% Xmedian=median(Xdata);
% % Xmedian=0;
% 
% figure;
% ax=subplot(1,3,1);
% % Principal Gaussian overbound (zero-mean)
% % Type | Ele. | Inflate  | alpha |
% % Ref  | 30-35| 1,   1.15|  0.7  |
% % Ref  | 60-65| ?        |  0.7  |
% % Urban| 30-35| 1,   2.2 |  0.7  |
% % Urban| 30-80| 1.2, 2   |  0.9  |
% gmm_dist_raw=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_raw,1,2.2) % inflate_ref30: (1,1.15); inflate_urban_30-80: (1.2,2); inflate_urban_30:
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,0.9); %ref30:0.9; ref60:0.7; urban_30-80:0.9
% [s1_list,s2_list]=YanFuncLib_Overbound_tmp.gen_s1_s2(x_lin,Xdata,gmm_dist,0,ax);
% plot(ax,x_lin,pdf_pgo,'g','LineWidth',2);
% 
% ax=subplot(1,3,2);
% % Principal Gaussian overbound (left)
% Xleft=Xdata(Xdata<Xmedian);
% Xleft_recon=[Xleft;2*Xmedian-Xleft;Xmedian];
% gmm_dist_left=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xleft_recon-mean(Xleft_recon));
% gmm_dist_left=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_left,1,1.5) % inflate: 1.1; inflate: 1.5
% [s1_list,s2_list]=YanFuncLib_Overbound_tmp.gen_s1_s2(x_lin,Xleft_recon,gmm_dist_left,mean(Xleft_recon),ax);
% [params_pgo_left, pdf_pgo_left, cdf_pgo_left]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xleft_recon-mean(Xleft_recon),x_lin,gmm_dist_left,0.7);
% plot(ax,x_lin+mean(Xleft_recon),pdf_pgo_left,'g','LineWidth',2);
% 
% ax=subplot(1,3,3);
% % Principal Gaussian overbound (right)
% Xright=Xdata(Xdata>Xmedian);
% Xright_recon=[Xright;2*Xmedian-Xright;Xmedian];
% gmm_dist_right=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xright_recon-mean(Xright_recon));
% gmm_dist_right=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_right,1,1.5) % inflate: 1.1;inflate: 1.5
% [s1_list,s2_list]=YanFuncLib_Overbound_tmp.gen_s1_s2(x_lin,Xright_recon,gmm_dist_right,mean(Xright_recon),ax);
% [params_pgo_right, pdf_pgo_right, cdf_pgo_right]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xright_recon-mean(Xright_recon),x_lin,gmm_dist_right,0.9);
% plot(ax,x_lin+mean(Xright_recon),pdf_pgo_right,'g','LineWidth',2);
% 
% % cdf
% figure;
% h1=plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
% hold on
% % Two step Gaussian
% [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound(Xdata,x_lin);
% h21=plot(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'r','LineWidth',2);
% h22=plot(x_lin(params.idx+1:end),cdf_left_tsgo(params.idx+1:end),'y','LineWidth',1);
% h23=plot(x_lin(1:params.idx),cdf_right_tsgo(1:params.idx),'y--','LineWidth',1);
% h24=plot(x_lin(params.idx+1:end),cdf_right_tsgo(params.idx+1:end),'r--','LineWidth',2);
% % Gaussian Pareto
% [params_gpo,pdf_gpo,cdf_gpo]=YanFuncLib_Overbound_tmp.Gaussian_Pareto_bound(Xdata,x_lin);
% h3=plot(x_lin,cdf_gpo,'b','LineWidth',2);
% h41=plot(x_lin(1:ceil(counts/2))+mean(Xleft_recon),cdf_pgo_left(1:ceil(counts/2)),'g','LineWidth',2);
% h42=plot(x_lin(ceil(counts/2)+1:end)+mean(Xright_recon),cdf_pgo_right(ceil(counts/2)+1:end),'g--','LineWidth',2);
% h5=plot(x_lin,cdf_pgo,'md-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % xline(Xmedian);
% % xline(0);
% xlabel('Error','FontSize',12);
% ylabel('CDF','FontSize',12);
% A = legend([h1,h21,h24,h3,h41,h42,h5],'Sample dist.',...
%     'Two-step Gaussian (left)','Two-step Gaussian (right)',....
%     'Gaussian Pareto','Paired Principal Gaussian (left)',...
%     'Paired Principal Gaussian (right)','Principal Gaussian');
% set(A,'FontSize',12)
% set(gca, 'FontSize', 12,'FontName', 'Times New Roman');
% 
% 
% % log scale cdf (left side)
% figure;
% h1=semilogy(x_lin_ecdf,ecdf_data,'ko','LineWidth',1,'MarkerSize', 4);
% hold on
% h21=semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'r','LineWidth',2);
% h22=semilogy(x_lin(params.idx+1:end),cdf_left_tsgo(params.idx+1:end),'y','LineWidth',1);
% h23=semilogy(x_lin(1:params.idx),cdf_right_tsgo(1:params.idx),'y--','LineWidth',1);
% h24=semilogy(x_lin(params.idx+1:end),cdf_right_tsgo(params.idx+1:end),'r--','LineWidth',2);
% 
% h3=semilogy(x_lin,cdf_gpo,'b','LineWidth',2);
% h41=semilogy(x_lin(1:ceil(counts/2))+mean(Xleft_recon),cdf_pgo_left(1:ceil(counts/2)),'g','LineWidth',2);
% h42=semilogy(x_lin(ceil(counts/2)+1:end)+mean(Xright_recon),cdf_pgo_right(ceil(counts/2)+1:end),'g--','LineWidth',2);
% h5=semilogy(x_lin,cdf_pgo,'md-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % xline(Xmedian);
% % xline(0);
% xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% xlabel('Error','FontSize',12);
% ylabel('CDF (log scale)','FontSize',12);
% A = legend([h1,h21,h3,h41,h5],'Sample dist.','Two-step Gaussian (left)','Gaussian Pareto','Paired Principal Gaussian (left)','Principal Gaussian');
% set(A,'FontSize',12)
% set(gca, 'FontSize', 12,'FontName', 'Times New Roman');
% 
% % log scale cdf (right side)
% figure;
% h1=semilogy(x_lin_ecdf,1-ecdf_data,'ko','LineWidth',1,'MarkerSize', 4);
% hold on
% h21=semilogy(x_lin(1:params.idx),1-cdf_left_tsgo(1:params.idx),'r--','LineWidth',2);
% h22=semilogy(x_lin(params.idx+1:end),1-cdf_left_tsgo(params.idx+1:end),'y','LineWidth',1);
% h23=semilogy(x_lin(1:params.idx),1-cdf_right_tsgo(1:params.idx),'y--','LineWidth',1);
% h24=semilogy(x_lin(params.idx+1:end),1-cdf_right_tsgo(params.idx+1:end),'r','LineWidth',2);
% 
% h3=semilogy(x_lin,1-cdf_gpo,'b','LineWidth',2);
% h41=semilogy(x_lin(1:ceil(counts/2))+mean(Xleft_recon),1-cdf_pgo_left(1:ceil(counts/2)),'g--','LineWidth',2);
% h42=semilogy(x_lin(ceil(counts/2)+1:end)+mean(Xright_recon),1-cdf_pgo_right(ceil(counts/2)+1:end),'g','LineWidth',2);
% h5=semilogy(x_lin,1-cdf_pgo,'md-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % xline(Xmedian);
% % xline(0);
% xlim([min(x_lin)*0.5,max(x_lin)*1.2])
% xlabel('Error','FontSize',12);
% ylabel('CCDF (log scale)','FontSize',12);
% A = legend([h1,h24,h3,h42,h5],'Sample dist.','Two-step Gaussian (right)','Gaussian Pareto','Paired Principal Gaussian (right)','Principal Gaussian');
% set(A,'FontSize',12)
% set(gca, 'FontSize', 12,'FontName', 'Times New Roman');
% 
% % qq plot
% figure;
% % define quantile range
% q_lin=linspace(0.001, 1-0.001, 10000);
% x_lin2=linspace(-3, 3, 10000); % solved: wrong use of griddedInterpolant
% cdf_norm=normcdf(x_lin2,0,1);
% ref_q=interp1(cdf_norm, x_lin2, q_lin, 'linear', 'extrap');
% data_source = interp1(ecdf_data, x_lin_ecdf, q_lin, 'linear', 'extrap');
% data_tsgo_left = interp1(cdf_left_tsgo, x_lin, q_lin, 'linear', 'extrap');
% data_tsgo_right = interp1(cdf_right_tsgo, x_lin, q_lin, 'linear', 'extrap');
% data_gpo = interp1(cdf_gpo, x_lin, q_lin, 'linear', 'extrap');
% cdf_pgo_con=[cdf_pgo_left(1:ceil(counts/2)),cdf_pgo_right(ceil(counts/2)+1:end)];
% data_pgo_con = interp1(cdf_pgo_con, x_lin+mean(Xleft_recon), q_lin, 'linear', 'extrap');
% data_pgo = interp1(cdf_pgo, x_lin, q_lin, 'linear', 'extrap');
% 
% h1=plot(data_source,ref_q);
% hold on
% h21=plot(data_tsgo_left,ref_q,'r','LineWidth',2);
% h24=plot(data_tsgo_right,ref_q,'m','LineWidth',2);
% h3=plot(data_gpo,ref_q,'b','LineWidth',2);
% h4=plot(data_pgo_con,ref_q,'g','LineWidth',2);
% h5=plot(data_pgo,ref_q,'y','LineWidth',2);
% yline(0);
% xlabel('Quantiles of error distribution','FontSize',12);
% ylabel('Standard normal quantile','FontSize',12);
% 
% A = legend([h1,h21,h24,h3,h4,h5],'sample dist.','two-step left','two-step right','Gaussian Pareto','Paired Principal Gaussian','Principal Gaussian');
% set(A,'FontSize',12)
% set(gca, 'FontSize', 12,'FontName', 'Times New Roman');

%% 20240102 model all ele_bin for Ref data
% seed=1234;
% gmm_cells=cell(0);
% tsgo_cells=cell(0);
% pgo_cells=cell(0);
% inflate_cells=cell(0);
% i=1;
% ele_start_list=15:5:75;
% for ele_start=15:5:75
%     try
%         % load Data
%         [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD('Data/mnav_zmp1_jan_20240105/mergedRefJan.mat',ele_start,5);
% %         [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD('Data/urban_dd_20240104/mergeurbandd.mat',30,50);% use all data for fitting
% %         [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD();
%         pdf_emp = ksdensity(Xdata,x_lin);
%         cdf_emp=cumtrapz(pdf_emp);
%         cdf_emp=cdf_emp*(x_lin(2)-x_lin(1));
%     
%         [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
%         counts=length(x_lin);
%         % fit gmm
%         gmm_dist_raw=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
%         % Two-step Gaussian overbound (zero-mean)
%         [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_zero(Xdata,x_lin);
%         param_tsgo = std_tsgo;
%         % Principal Gaussian overbound (zero-mean)
%         inflate_core=1; inflate_tail=1; thr=0.7;
%         gmm_dist_inflate=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_raw,inflate_core,inflate_tail); % inflate
%         [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist_inflate,thr);
%         % store 
%         gmm_cells{i}=gmm_dist_raw;
%         tsgo_cells{i}=param_tsgo;
%         pgo_cells{i}=params_pgo;
%         inflate_cells{1,1}=inflate_core;
%         inflate_cells{2,1}=inflate_tail;
%         inflate_cells{3,1}=thr;
%         
% %         close all
% %         % show pdf
% %         figure
% %         subplot(1,2,1)
% %         % plot(x_lin,pdf_data,'k','LineWidth',2);
% %         histogram(Xdata,'normalization','pdf')
% %         hold on
% %         plot(x_lin,pdf_emp,'k--','LineWidth',2);
% %         plot(x_lin,pdf_tsgo,'r','LineWidth',2);
% %         plot(x_lin,pdf_pgo,'g','LineWidth',2);
% %         xlabel('Error','FontSize',12);
% %         ylabel('PDF','FontSize',12);
% %         A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
% %         set(A,'FontSize',12)
% %         % show cdf
% %         subplot(1,2,2)
% %         plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
% %         hold on
% %         plot(x_lin,cdf_emp,'k--','LineWidth',2);
% %         plot(x_lin,cdf_tsgo,'r','LineWidth',2);
% %         plot(x_lin,cdf_pgo,'g','LineWidth',2);
% %         xlabel('Error','FontSize',12);
% %         ylabel('CDF','FontSize',12);
% %         A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
% %         set(A,'FontSize',12)
% %         
% %         % log scale cdf (left side)
% %         figure;
% %         h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4);
% %         hold on
% %         h21=semilogy(x_lin,cdf_tsgo,'g','LineWidth',2);
% %         h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% %         xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% %         xlabel('Error (m)');
% %         ylabel('CDF (log scale)');
% %         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% %         A = legend([h1,h21,h5],'Sample dist.','Gaussian','Principal Gaussian');
% %         set(A,'FontSize',13.5)
% %         grid on
% %         
% %         
% %         % log scale cdf (right side)
% %         figure;
% %         h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 4);
% %         hold on
% %         h24=semilogy(x_lin,1-cdf_tsgo,'g','LineWidth',2);
% %         h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% %         xlim([min(x_lin)*0.5,max(x_lin)*1.2])
% %         xlabel('Error (m)');
% %         ylabel('CCDF (log scale)');
% %         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% %         A = legend([h1,h24,h5],'Sample dist.','Gaussian','Principal Gaussian');
% %         set(A,'FontSize',13.5)
% %         grid on
% %         aa=0;
%     catch
%         aa=0;
%     end
%     i=i+1;
% end
% save('ref_overbounding',"gmm_cells","tsgo_cells","pgo_cells","ele_start_list","inflate_cells")

%% 20240102 WLS+detection - urban
% load('Data/urban_dd_20240104/mergeurbandd.mat');
% load('urban_overbounding.mat');
% tData_sorted = sortrows(mergedurbandd,1);
% all_epoches = sort(unique(tData_sorted.datetime));
% ele_start=ele_start_list(1);
% ele_step = ele_start_list(2)-ele_start_list(1);
% % init detection matrix
% detec_mat=zeros(length(all_epoches),3);
% time_jk_list=[];
% time_ss_list=[];
% for i=1:length(all_epoches)
%     epoch = all_epoches(i);
%     % select epoch
%     filter_date= (tData_sorted.datetime==epoch);
%     Xdata_raw=tData_sorted(filter_date,:);
%     % select eligeble ele and snr for WLS
%     % (yihan's filter method)
%     filter_ele = (Xdata_raw.U2I_Elevation>30 & Xdata_raw.U2M_Elevation>30 & Xdata_raw.R2I_Elevation>30 & Xdata_raw.R2M_Elevation>30);
%     filter_snr = (Xdata_raw.U2I_SNR>40 & Xdata_raw.U2M_SNR>40 & Xdata_raw.R2I_SNR>40 & Xdata_raw.R2M_SNR>40);
%     Xdata=Xdata_raw(filter_ele&filter_snr,:);
%     % meas
%     meas = Xdata.double_differenced_pseudorange;
%     % find ele bin
%     ele_list = Xdata.U2I_Elevation;
%     bin_list = ceil((ele_list-ele_start)/ele_step);
%     
%     % sat position
%     m_sv_pos = [Xdata.master_x,Xdata.master_y,Xdata.master_z];
%     s_sv_pos = [Xdata.target_x,Xdata.target_y,Xdata.target_z];
%     % other info.
%     init_state=zeros(3,1);
%     GTusr_xyz = [-2418235.676841056, 5386096.899553243, 2404950.408609563];
%     ref_xyz = [-2414266.9197, 5386768.9868, 2407460.0314];
% %     % WLS solution
% %     if size(s_sv_pos,1)>=3
% %         [eWLSSolution,prn_res,ErrorECEF,G,eDeltaPos,eDeltaPr] = WeightedLeastSquareDD(GTusr_xyz, init_state, meas, meas_std, m_sv_pos,s_sv_pos,ref_xyz);
% %         error_list(i)=ErrorECEF;
% %     else
% %         error_list(i)=-1;
% %     end
% %     time_list(i)=Xdata.gps_week(1) * 604800.0 + Xdata.gps_Sec(1) + 315964800.0 + 19.0;
% 
%     % detection and positioning
%     if size(s_sv_pos,1)>=4
%         % sigma construction 
%         tsgo_meas_std=[];
%         pgo_meas_std=[];
%         pgo_current_cells=cell(0);
%         for ii=1:length(bin_list)
%             bin=bin_list(ii);
%             try
%                 % std of two-step Gaussian overbound
%                 tsgo_params = tsgo_cells{bin};
%                 tsgo_meas_std(ii,1)=tsgo_params;
%                 % sigma of PGO
%                 pgo_params = pgo_cells{bin};
%                 pgo_meas_std(ii,1)=sqrt(pgo_params.p1*pgo_params.sigma1+(1-pgo_params.p1)*pgo_params.sigma2);
%                 pgo_current_cells{ii}=pgo_params;
%             catch
%                 tsgo_meas_std(ii,1) = 1;
%                 pgo_meas_std(ii,1) = 1;
%             end
%         end
%         % simulate failure
%         inject_bias=20;
%         meas(1)=meas(1)+inject_bias;
%         % detection and positioning
%         t_start=tic;
%         is_detect_jackknife = jackknife_detector(GTusr_xyz,meas,pgo_meas_std,m_sv_pos,s_sv_pos,ref_xyz,pgo_current_cells,"PGO");
%         time_jk_list(end+1)=toc(t_start);
%         t_start2=tic;
%         is_detect_ss        = ss_detector(GTusr_xyz,meas,tsgo_meas_std,m_sv_pos,s_sv_pos,ref_xyz);
%         time_ss_list(end+1)=toc(t_start2);
%         detec_mat(i,:)=[is_detect_jackknife,is_detect_ss,0];
%     else
%         % invalid detection
%         detec_mat(i,:)=[0,0,-1];
%     end
% end
% figure
% scatter(1:length(all_epoches),detec_mat(:,1))
% hold on
% scatter(1:length(all_epoches),detec_mat(:,2),'*')
% legend('Jackknife','Solution separation')
% 
% total_epochs=size(detec_mat,1);
% invalid_epochs=abs(sum(detec_mat(:,3)));
% Pdec_jackknife = sum(detec_mat(:,1))/(total_epochs-invalid_epochs)
% Pdec_ss = sum(detec_mat(:,2))/(total_epochs-invalid_epochs)

%% 20240104 Re-Check the calculation of DD error from yihan
% load('Data/urban_dd_20240104/mergeurbandd.mat');
% m_sv_pos = [mergedurbandd.master_x,mergedurbandd.master_y,mergedurbandd.master_z];
% s_sv_pos = [mergedurbandd.target_x,mergedurbandd.target_y,mergedurbandd.target_z];
% % ref_xyz = [-2414266.9197, 5386768.9868, 2407460.0314];
% % GTusr_xyz = [-2418235.676841056, 5386096.899553243, 2404950.408609563];
% ref_xyz =[-254127.011,-4531607.048,4466509.757]; % MNAV ref station
% GTusr_xyz =[-249978.306,-4539297.200,4458954.757]; % ZMP1 ref station
% true_geo = (sqrt(sum((m_sv_pos-ref_xyz).*(m_sv_pos-ref_xyz),2)) - sqrt(sum((m_sv_pos-GTusr_xyz).*(m_sv_pos-GTusr_xyz),2))) ....
%             -(sqrt(sum((s_sv_pos-ref_xyz).*(s_sv_pos-ref_xyz),2)) - sqrt(sum((s_sv_pos-GTusr_xyz).*(s_sv_pos-GTusr_xyz),2)));
% 
% rho_u2m = mergedurbandd.user2master_pseudorange;
% rho_u2s = mergedurbandd.user2target_pseudorange;
% rho_r2m = mergedurbandd.ref2master_pseudorange;
% rho_r2s = mergedurbandd.ref2target_pseudorange;
% dd_meas = (rho_u2m-rho_r2m) - (rho_u2s-rho_r2s);
% dd_meas = dd_meas*(-1);
% 
% dd_error = true_geo-dd_meas;
% 
% compare_data = [mergedurbandd.double_differenced_true_geometric_range - true_geo, ...
%                mergedurbandd.double_differenced_pseudorange - dd_meas,...
%                mergedurbandd.doubledifferenced_pseudorange_error - dd_error];

%% 20240106 WLS+detection - ref
% load('Data/mnav_zmp1_jan_20240105/mergedRefJan.mat');
% load('ref_overbounding_correction.mat');
% mergedRefJan_select = mergedRefJan(mergedRefJan.datetime>="2020-01-01" & mergedRefJan.datetime<"2020-01-02",:);
% tData_sorted = sortrows(mergedRefJan_select,1);
% all_epoches = sort(unique(tData_sorted.datetime));
% ele_start=ele_start_list(1);
% ele_step = ele_start_list(2)-ele_start_list(1);
% for inject_bias_set=0:10
%     inject_bias_set
%     % init detection matrix
%     detec_mat=zeros(length(all_epoches),3);
%     time_jk_list=[];
%     time_ss_list=[];
%     % 1914
%     for i=1:length(all_epoches)
%         epoch = all_epoches(i);
%         % select epoch
%         filter_date= (tData_sorted.datetime==epoch);
%         Xdata_raw=tData_sorted(filter_date,:);
%         % select eligeble ele and snr for WLS (do not apply selection for reference station data)
%         Xdata=Xdata_raw;
%         % meas
%         meas = Xdata.double_differenced_pseudorange;
%         % find ele bin
%         ele_list = Xdata.U2I_Elevation;
%         bin_list = ceil((ele_list-ele_start)/ele_step);
% 
%         % sat position
%         m_sv_pos = [Xdata.master_x,Xdata.master_y,Xdata.master_z];
%         s_sv_pos = [Xdata.target_x,Xdata.target_y,Xdata.target_z];
%         % other info.
%         init_state=zeros(3,1);
%         ref_xyz =[-254127.011,-4531607.048,4466509.757]; % MNAV ref station
%         GTusr_xyz =[-249978.306,-4539297.200,4458954.757]; % ZMP1 ref station
%     %     % WLS solution
%     %     if size(s_sv_pos,1)>=3
%     %         [eWLSSolution,prn_res,ErrorECEF,G,eDeltaPos,eDeltaPr] = WeightedLeastSquareDD(GTusr_xyz, init_state, meas, meas_std, m_sv_pos,s_sv_pos,ref_xyz);
%     %         error_list(i)=ErrorECEF;
%     %     else
%     %         error_list(i)=-1;
%     %     end
%     %     time_list(i)=Xdata.gps_week(1) * 604800.0 + Xdata.gps_Sec(1) + 315964800.0 + 19.0;
% 
%         % detection and positioning
%         if i==1914
%             detec_mat(i,:)=[-2,-2,-1]; % unsolved problem?
%             time_jk_list(end+1)=-1;
%             time_ss_list(end+1)=-1;
%             continue
%         end
%         if size(s_sv_pos,1)>=4 
%             % sigma construction 
%             tsgo_meas_std=[];
%             pgo_meas_std=[];
%             pgo_current_cells=cell(0);
%             for ii=1:length(bin_list)
%                 bin=bin_list(ii);
%                 try
%                     % sigma of two-step Gaussian overbound
%                     tsgo_params = tsgo_cells{bin};
%                     tsgo_meas_std(ii,1)=tsgo_params;
%                     % sigma of PGO
%                     pgo_params = pgo_cells{bin};
%                     pgo_meas_std(ii,1)=sqrt(pgo_params.p1*pgo_params.sigma1+(1-pgo_params.p1)*pgo_params.sigma2);
%                     pgo_current_cells{ii}=pgo_params;
%                 catch
%                     tsgo_meas_std(ii,1) = 1;
%                     pgo_meas_std(ii,1) = 1;
%                 end
%             end
%             % simulate failure
%             inject_bias=inject_bias_set;
%             meas(1)=meas(1)+inject_bias;
%             % detection and positioning
%             t_start=tic;
%             is_detect_jackknife = jackknife_detector(GTusr_xyz,meas,pgo_meas_std,m_sv_pos,s_sv_pos,ref_xyz,pgo_current_cells,"PGO");
%             time_jk_list(end+1)=toc(t_start);
%             t_start2=tic;
%             is_detect_ss        = ss_detector(GTusr_xyz,meas,tsgo_meas_std,m_sv_pos,s_sv_pos,ref_xyz);
%             time_ss_list(end+1)=toc(t_start2);
%             detec_mat(i,:)=[is_detect_jackknife,is_detect_ss,0];
%         else
%             % invalid detection
%             detec_mat(i,:)=[0,0,-1];
%             time_jk_list(end+1)=-1;
%             time_ss_list(end+1)=-1;
%         end
%     end
% %     figure
% %     scatter(1:length(all_epoches),detec_mat(:,1))
% %     hold on
% %     scatter(1:length(all_epoches),detec_mat(:,2),'*')
% %     legend('Jackknife','Solution separation')
% 
%     total_epochs=size(detec_mat,1);
%     invalid_epochs=abs(sum(detec_mat(:,3)));
%     % delete invalid epochs
%     detec_mat_valid=detec_mat((detec_mat(:,3)~=-1),:);
%     % detect rate
%     valid_epochs=size(detec_mat_valid,1);
%     Pdec_jackknife = sum(detec_mat(:,1))/valid_epochs
%     Pdec_ss = sum(detec_mat(:,2))/valid_epochs
%     % save results
%     save(['dd_ref_correct_bias',num2str(inject_bias)],"total_epochs","invalid_epochs","valid_epochs","detec_mat","detec_mat_valid","Pdec_jackknife","Pdec_ss","time_jk_list","time_ss_list");
% end

%% 20240108 visualize detection performance of SS and JK 
% JK_list_show=zeros(10,1);
% SS_list_show=zeros(10,1);
% for k=1:10
%     file=['dd_ref_correct_bias',num2str(k)];
%     load(file)
% %     % delete invalid epochs
% %     detec_mat_valid=detec_mat((detec_mat(:,3)~=-1),:);
% %     % detect rate
% %     valid_epochs=size(detec_mat_valid,1);
% %     Pdec_jackknife = sum(detec_mat(:,1))/valid_epochs;
% %     Pdec_ss = sum(detec_mat(:,2))/valid_epochs;
%     JK_list_show(k)=Pdec_jackknife;
%     SS_list_show(k)=Pdec_ss;
% end
% plot(JK_list_show*100,'b-o')
% hold on
% plot(SS_list_show*100,'r-*')


%% 20240114 model all ele_bin for Ref CHTI
% seed=1234;
% gmm_cells=cell(0);
% tsgo_cells=cell(0);
% pgo_cells=cell(0);
% inflate_cells=cell(0);
% i=1;
% ele_start_list=15:5:85;
% for ele_start=80:5:85
%     try
%         % load Data
%         [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefSPP('Data/cors_CHTI_Jan/mergedCHTIJan_exd.mat',ele_start,5);
%         pdf_emp = ksdensity(Xdata,x_lin);
%         cdf_emp=cumtrapz(pdf_emp);
%         cdf_emp=cdf_emp*(x_lin(2)-x_lin(1));
%     
%         [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
%         counts=length(x_lin);
%         % fit gmm
%         gmm_dist_raw=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
%         % Two-step Gaussian overbound (zero-mean)
%         [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_zero(Xdata,x_lin);
%         param_tsgo = std_tsgo;
%         % Principal Gaussian overbound (zero-mean)
%         inflate_core=1; inflate_tail=1; thr=0.7;
%         gmm_dist_inflate=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_raw,inflate_core,inflate_tail); % inflate
%         [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist_inflate,thr);
%         % store 
%         gmm_cells{i}=gmm_dist_raw;
%         tsgo_cells{i}=param_tsgo;
%         pgo_cells{i}=params_pgo;
%         inflate_cells{1,i}=inflate_core;
%         inflate_cells{2,i}=inflate_tail;
%         inflate_cells{3,i}=thr;
%         
% %         close all
% %         % show pdf
% %         figure
% %         subplot(1,2,1)
% %         % plot(x_lin,pdf_data,'k','LineWidth',2);
% %         histogram(Xdata,'normalization','pdf')
% %         hold on
% %         plot(x_lin,pdf_emp,'k--','LineWidth',2);
% %         plot(x_lin,pdf_tsgo,'r','LineWidth',2);
% %         plot(x_lin,pdf_pgo,'g','LineWidth',2);
% %         xlabel('Error','FontSize',12);
% %         ylabel('PDF','FontSize',12);
% %         A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
% %         set(A,'FontSize',12)
% %         % show cdf
% %         subplot(1,2,2)
% %         plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
% %         hold on
% %         plot(x_lin,cdf_emp,'k--','LineWidth',2);
% %         plot(x_lin,cdf_tsgo,'r','LineWidth',2);
% %         plot(x_lin,cdf_pgo,'g','LineWidth',2);
% %         xlabel('Error','FontSize',12);
% %         ylabel('CDF','FontSize',12);
% %         A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
% %         set(A,'FontSize',12)
% %         
% %         % log scale cdf (left side)
% %         figure;
% %         h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4);
% %         hold on
% %         h21=semilogy(x_lin,cdf_tsgo,'g','LineWidth',2);
% %         h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% %         xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% %         xlabel('Error (m)');
% %         ylabel('CDF (log scale)');
% %         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% %         A = legend([h1,h21,h5],'Sample dist.','Gaussian','Principal Gaussian');
% %         set(A,'FontSize',13.5)
% %         grid on
% %         
% %         
% %         % log scale cdf (right side)
% %         figure;
% %         h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 4);
% %         hold on
% %         h24=semilogy(x_lin,1-cdf_tsgo,'g','LineWidth',2);
% %         h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% %         xlim([min(x_lin)*0.5,max(x_lin)*1.2])
% %         xlabel('Error (m)');
% %         ylabel('CCDF (log scale)');
% %         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% %         A = legend([h1,h24,h5],'Sample dist.','Gaussian','Principal Gaussian');
% %         set(A,'FontSize',13.5)
% %         grid on
% %         aa=0;
%     catch
%         aa=0;
%     end
%     i=i+1;
% end
% save('CHTI_overbounding',"gmm_cells","tsgo_cells","pgo_cells","ele_start_list","inflate_cells")

%% 20240114 WLS+detection - ref CHTI
% load('Data/cors_CHTI_Jan/mergedCHTIJan_exd.mat');   
% load('CHTI_overbounding_correction.mat');
% % % 2023-01-25
% % mergedRefJan_select = mergedCHTIJan_exd(mergedCHTIJan_exd.gps_time>=1674604800 & mergedCHTIJan_exd.gps_time<1674691170,:);
% % 2023-01-28
% mergedRefJan_select = mergedCHTIJan_exd(mergedCHTIJan_exd.gps_time>=1674864000 & mergedCHTIJan_exd.gps_time<1674950370,:);
% tData_sorted = sortrows(mergedRefJan_select,1);
% all_epoches = sort(unique(tData_sorted.gps_time));
% ele_start=ele_start_list(1);
% ele_step = ele_start_list(2)-ele_start_list(1);
% 
% inject_bias_set=0
% % init detection matrix
% detec_mat=zeros(length(all_epoches),3);
% time_jk_list=[];
% time_ss_list=[];
% % 
% for i=1:length(all_epoches)
%     epoch = all_epoches(i);
%     % select epoch
%     filter_date= (tData_sorted.gps_time==epoch);
%     % select ele
%     filter_ele=(tData_sorted.ele>=15*3.1415/180);
%     Xdata_raw=tData_sorted(filter_date & filter_ele,:);
%     % select eligeble ele and snr for WLS (do not apply selection for reference station data)
%     Xdata=Xdata_raw;
%     % meas
%     meas = Xdata.cor_pseudo;
%     % find ele bin
%     ele_list = Xdata.ele * 180/3.1415; % rad to deg
%     bin_list = ceil((ele_list-ele_start)/ele_step);
% 
%     % sat position
%     m_sv_pos = [Xdata.sat_x,Xdata.sat_y,Xdata.sat_z];
%     % other info.
%     init_state=zeros(4,1); % x,y,z,dt
%     GTusr_xyz =[-4607855.993,-272375.743,-4386954.429]; % CHTI station
% 
%     % detection and positioning
% %         if i==1914
% %             detec_mat(i,:)=[-2,-2,-1]; % unsolved problem?
% %             time_jk_list(end+1)=-1;
% %             time_ss_list(end+1)=-1;
% %             continue
% %         end
%     if size(m_sv_pos,1)>=4 
%         % sigma construction 
%         tsgo_meas_std=[];
%         pgo_meas_std=[];
%         pgo_current_cells=cell(0);
%         for ii=1:length(bin_list)
%             bin=bin_list(ii);
%             try
%                 % sigma of two-step Gaussian overbound
%                 tsgo_params = tsgo_cells{bin};
%                 tsgo_meas_std(ii,1)=tsgo_params;
%                 % sigma of PGO
%                 pgo_params = pgo_cells{bin};
%                 pgo_meas_std(ii,1)=sqrt(pgo_params.p1*pgo_params.sigma1+(1-pgo_params.p1)*pgo_params.sigma2);
%                 pgo_current_cells{ii}=pgo_params;
%             catch
%                 tsgo_meas_std(ii,1) = 1;
%                 pgo_meas_std(ii,1) = 1;
%             end
%         end
%         % simulate failure
%         inject_bias=inject_bias_set;
%         meas(1)=meas(1)+inject_bias;
%         % detection and positioning
%         t_start=tic;
%         is_detect_jackknife = jackknife_detector_SPP(GTusr_xyz,meas,pgo_meas_std,m_sv_pos,pgo_current_cells,"PGO");
%         time_jk_list(end+1)=toc(t_start);
%         t_start2=tic;
%         is_detect_ss        =ss_detector_SPP(GTusr_xyz,meas,tsgo_meas_std,m_sv_pos);
%         time_ss_list(end+1)=toc(t_start2);
%         detec_mat(i,:)=[is_detect_jackknife,is_detect_ss,0];
%     else
%         % invalid detection
%         detec_mat(i,:)=[0,0,-1];
%         time_jk_list(end+1)=-1;
%         time_ss_list(end+1)=-1;
%     end
% end
% figure
% scatter(1:length(all_epoches),detec_mat(:,1))
% hold on
% scatter(1:length(all_epoches),detec_mat(:,2),'*')
% legend('Jackknife','Solution separation')
% 
% total_epochs=size(detec_mat,1);
% invalid_epochs=abs(sum(detec_mat(:,3)));
% % delete invalid epochs
% detec_mat_valid=detec_mat((detec_mat(:,3)~=-1),:);
% % detect rate
% valid_epochs=size(detec_mat_valid,1);
% Pdec_jackknife = sum(detec_mat(:,1))/valid_epochs
% Pdec_ss = sum(detec_mat(:,2))/valid_epochs
% aa=0;
% % save results
% save('CHTI_Jan28_detection',"total_epochs","invalid_epochs","valid_epochs","detec_mat","detec_mat_valid","Pdec_jackknife","Pdec_ss","time_jk_list","time_ss_list");
% save('CHTI_Jan28_detection_raw')

%% 2024-01-16 visualization of Jan 28 detection
load('CHTI_Jan28_detection_raw')
k=1;
for i=1:length(all_epoches)
    epoch = all_epoches(i);
    % select epoch
    filter_date= (tData_sorted.gps_time==epoch);
    % select ele
    filter_ele=(tData_sorted.ele>=15*3.1415/180);
    Xdata_raw=tData_sorted(filter_date & filter_ele,:);
    % visibility of PRN 1
    if ismember(1, Xdata_raw.sat_id)
        visi_PRN1_list(i)=1;
        prn1_data = Xdata_raw(Xdata_raw.sat_id==1,:);
        prn1_meas_ele_list(k)=prn1_data.ele*180/3.1415;
        prn1_meas_error_list(k)=prn1_data.cor_pseudo-prn1_data.range;
        prn1_meas_time_list(k)=i;
        k=k+1;
    else
        visi_PRN1_list(i)=0;
    end
end
% % figure
% % scatter(1:length(all_epoches),visi_PRN1_list)
% figure
% scatter(1:length(all_epoches),detec_mat(:,1),'k','filled')
% hold on
% % Shade the area: PRN1 is visible
% xshade = [689:987,flip(689:987)]; % x coordinates for the shaded area
% yshade = [ones(1,987-689+1),zeros(1,987-689+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none');
% % Shade the area: PRN1 is visible
% xshade = [1697:2206,flip(1697:2206)]; % x coordinates for the shaded area
% yshade = [ones(1,2206-1697+1),zeros(1,2206-1697+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none','HandleVisibility','off')
% % Shade the area: fault period
% xshade = [1805:2400,flip(1805:2400)]; % x coordinates for the shaded area
% yshade = [ones(1,2400-1805+1),zeros(1,2400-1805+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'r', 'FaceAlpha', 0.2,'EdgeColor','none');
% legend('Detection state','PRN1 is visible','PRN1 fault period')
% title('Detection states of Jackknife test detector on Jan. 28')
% 
% figure
% scatter(1:length(all_epoches),detec_mat(:,2),'k','filled')
% hold on
% % Shade the area: PRN1 is visible
% xshade = [689:987,flip(689:987)]; % x coordinates for the shaded area
% yshade = [ones(1,987-689+1),zeros(1,987-689+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none');
% % Shade the area: PRN1 is visible
% xshade = [1697:2206,flip(1697:2206)]; % x coordinates for the shaded area
% yshade = [ones(1,2206-1697+1),zeros(1,2206-1697+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none','HandleVisibility','off')
% % Shade the area: fault period
% xshade = [1805:2400,flip(1805:2400)]; % x coordinates for the shaded area
% yshade = [ones(1,2400-1805+1),zeros(1,2400-1805+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'r', 'FaceAlpha', 0.2,'EdgeColor','none');
% legend('Detection state','PRN1 is visible','PRN1 fault period')
% title('Detection states of solution separation detector on Jan. 28')


figure
scatter(prn1_meas_time_list,prn1_meas_ele_list,'HandleVisibility','off')
hold on
% Shade the area: PRN1 is visible
xshade = [689:987,flip(689:987)]; % x coordinates for the shaded area
yshade = [90*ones(1,987-689+1),zeros(1,987-689+1)]; % y coordinates for the shaded area
fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none');
% Shade the area: PRN1 is visible
xshade = [1697:2206,flip(1697:2206)]; % x coordinates for the shaded area
yshade = [90*ones(1,2206-1697+1),zeros(1,2206-1697+1)]; % y coordinates for the shaded area
fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none','HandleVisibility','off')
% Shade the area: fault period
xshade = [1805:2400,flip(1805:2400)]; % x coordinates for the shaded area
yshade = [90*ones(1,2400-1805+1),zeros(1,2400-1805+1)]; % y coordinates for the shaded area
fill(xshade, yshade, 'r', 'FaceAlpha', 0.2,'EdgeColor','none');
legend('PRN1 is visible','PRN1 fault period')
title('Elevation angle of PRN 1 on Jan. 28')

% % 'true' measurement error of PRN1 (actually, the receiver clock bias correction 
% % is not very trustable when producing cor_pseudo; therefore, this method might be misleading)
% figure
% scatter(prn1_meas_time_list,prn1_meas_error_list,'HandleVisibility','off')
% hold on
% % Shade the area: PRN1 is visible
% xshade = [689:987,flip(689:987)]; % x coordinates for the shaded area
% yshade = [ones(1,987-689+1),zeros(1,987-689+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none');
% % Shade the area: PRN1 is visible
% xshade = [1697:2206,flip(1697:2206)]; % x coordinates for the shaded area
% yshade = [ones(1,2206-1697+1),zeros(1,2206-1697+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'g', 'FaceAlpha', 0.3,'EdgeColor','none','HandleVisibility','off')
% % Shade the area: fault period
% xshade = [1805:2400,flip(1805:2400)]; % x coordinates for the shaded area
% yshade = [ones(1,2400-1805+1),zeros(1,2400-1805+1)]; % y coordinates for the shaded area
% fill(xshade, yshade, 'r', 'FaceAlpha', 0.2,'EdgeColor','none');
% legend('PRN1 is visible','PRN1 fault period')
% title('Measurement error of PRN 1 on Jan. 28')

function result = PGO_variance(pgo_params)
    k=pgo_params.k;
    cc=pgo_params.cc;
    p1=pgo_params.p1;
    std1=sqrt(pgo_params.sigma1);
    std2=sqrt(pgo_params.sigma2);
    xR2p = pgo_params.xR2p;
    pi=3.141596;
    % core part
    core = (2/3)*cc*xR2p^3 + ...
           std1*p1*(std1*erf(xR2p/(sqrt(2)*std1))-...
           sqrt(2/pi)*xR2p*exp(-xR2p^2/(2*std1^2)));
    
    tail = -(k+1)*(p1-1)*...
           (sqrt(pi/2)*std2^3*erfc(xR2p/(sqrt(2)*std2))+...
           std2^2*xR2p*exp(-xR2p^2/(2*std2^2)))/(sqrt(2*pi)*std2);
    tail = tail*2;
    result = core + tail;
end

function is_detect=jackknife_detector(GTusr_xyz,meas,meas_std,m_sv_pos,s_sv_pos,ref_xyz,pgo_current_cells,ob_type)
    is_detect=false;
    init_state=zeros(3,1);
    rho=-meas; % here is yihan's wrong operation, I need to correct it
    n=length(rho);
    Sigma=diag(meas_std.^2);
    W=inv(Sigma);
     % all-in-view solution
    [eWLSSolution,prn_res,ErrorECEF,H,eDeltaPos,eDeltaPr] = WeightedLeastSquareDD(GTusr_xyz, init_state, meas, meas_std, m_sv_pos,s_sv_pos,ref_xyz);
    S=inv(H'*W*H)*H'*W;
    
    % subsolution
    for i=1:n
        meas_sub=meas; meas_sub(i,:)=[];
        meas_std_sub=meas_std; meas_std_sub(i,:)=[];
        m_sv_pos_sub=m_sv_pos; m_sv_pos_sub(i,:)=[];
        s_sv_pos_sub=s_sv_pos; s_sv_pos_sub(i,:)=[];
        [eWLSSolution_sub,prn_res_sub,ErrorECEF3d_sub,H_sub,eDeltaPos_sub,eDeltaPr_sub]=WeightedLeastSquareDD(GTusr_xyz, init_state, meas_sub, meas_std_sub, m_sv_pos_sub,s_sv_pos_sub,ref_xyz);
        
        % construct los vector for i-th meas
        m_dGeoDistance = Euc_dis(m_sv_pos(i,:),eWLSSolution_sub');
        s_dGeoDistance = Euc_dis(s_sv_pos(i,:),eWLSSolution_sub');
        h_i =  [(eWLSSolution_sub(1)-m_sv_pos(i,1))/m_dGeoDistance - (eWLSSolution_sub(1)-s_sv_pos(i,1))/s_dGeoDistance, ...
                (eWLSSolution_sub(2)-m_sv_pos(i,2))/m_dGeoDistance - (eWLSSolution_sub(2)-s_sv_pos(i,2))/s_dGeoDistance, ...]
                (eWLSSolution_sub(3)-m_sv_pos(i,3))/m_dGeoDistance - (eWLSSolution_sub(3)-s_sv_pos(i,3))/s_dGeoDistance];

        % predicted i-th meas
        eDeltaPr_i_hat = h_i*eDeltaPos_sub;
        % test statistic
        rho_0=(Euc_dis(m_sv_pos(i,:),eWLSSolution_sub') - Euc_dis(m_sv_pos(i,:),ref_xyz))...
               -(Euc_dis(s_sv_pos(i,:),eWLSSolution_sub') - Euc_dis(s_sv_pos(i,:),ref_xyz));
        % note: the sign is different from paper, because the H matrix is
        % defined different with paper. Both forms are correct
        t_i = (rho(i)-rho_0) - eDeltaPr_i_hat;
        
        % deriving distribution of Jacknife residual
        Sigma_sub=diag(meas_std_sub.^2);
        W_sub=inv(Sigma_sub);
        S_sub=inv(H_sub'*W_sub*H_sub)*H_sub'*W_sub;
        S_subEx=[S_sub(:,1:i-1), zeros(3,1), S_sub(:,i:end)];
        if ob_type=="Gaussian"
            % null distribution of t_i under Gaussain overbound
            Sigma_t_i = h_i*S_subEx*Sigma*S_subEx'*h_i'+meas_std(i)^2;
            % threhold
            thr_i=sqrt(Sigma_t_i)*norminv(1-0.05/(2*n));
        elseif ob_type=="PGO"
            % extend t_i to non-Gaussian overbound
            H_rec = [H_sub(1:i-1,:); h_i; H_sub(i:end,:)];
            P_rec = H_rec*S_subEx;
            IsP_rec = eye(n)-P_rec;
            Isp_rec = IsP_rec(i,:); % i-th row of IsP_rec
            % set definition domain of the t_i
            x_scale=-10:0.01:10;
            scale_list=Isp_rec';
            PHMI=0.05/n;
            % solve distribution and critical value of t_i
            [PL_fake,~,fft_time_all]=YanFuncLib_Overbound_tmp.cal_PL_ex(scale_list,x_scale,[],pgo_current_cells,PHMI);
            % threhold
            thr_i=abs(PL_fake);
        end
        % detection
        if abs(t_i)>thr_i
            is_detect=true;
        end
    end  
end

function is_detect=jackknife_detector_SPP(GTusr_xyz,meas,meas_std,sv_pos,pgo_current_cells,ob_type)
    is_detect=false;
    init_state=zeros(4,1);
    rho=meas;
    n=length(rho);
    Sigma=diag(meas_std.^2);
    W=inv(Sigma);
     % all-in-view solution
    [eWLSSolution,prn_res,ErrorECEF,H,eDeltaPos,eDeltaPr] = WeightedLeastSquare(GTusr_xyz, init_state, meas, meas_std, sv_pos);
    S=inv(H'*W*H)*H'*W;
    
    % subsolution
    for i=1:n
        meas_sub=meas; meas_sub(i,:)=[];
        meas_std_sub=meas_std; meas_std_sub(i,:)=[];
        sv_pos_sub=sv_pos; sv_pos_sub(i,:)=[];
        [eWLSSolution_sub,prn_res_sub,ErrorECEF3d_sub,H_sub,eDeltaPos_sub,eDeltaPr_sub]=WeightedLeastSquare(GTusr_xyz, init_state, meas_sub, meas_std_sub, sv_pos_sub);
        
        % construct los vector for i-th meas
        dGeoDistance = Euc_dis(sv_pos(i,:),eWLSSolution_sub(1:3)');
        h_i =  [ - (sv_pos(i,1)-eWLSSolution_sub(1))/dGeoDistance, ...
                 - (sv_pos(i,2)-eWLSSolution_sub(2))/dGeoDistance, ...]
                 - (sv_pos(i,3)-eWLSSolution_sub(3))/dGeoDistance,...
                 1];

        % predicted i-th meas
        eDeltaPr_i_hat = h_i*eDeltaPos_sub;
        % test statistic
        rho_0 = Euc_dis(sv_pos(i,:),eWLSSolution_sub(1:3)')+eWLSSolution_sub(4);
        % note: the sign is different from paper, because the H matrix is
        % defined different with paper. Both forms are correct
        t_i = (rho(i)-rho_0) - eDeltaPr_i_hat;
        
        % deriving distribution of Jacknife residual
        Sigma_sub=diag(meas_std_sub.^2);
        W_sub=inv(Sigma_sub);
        S_sub=inv(H_sub'*W_sub*H_sub)*H_sub'*W_sub;
        S_subEx=[S_sub(:,1:i-1), zeros(4,1), S_sub(:,i:end)];
        if ob_type=="Gaussian"
            % null distribution of t_i under Gaussain overbound
            Sigma_t_i = h_i*S_subEx*Sigma*S_subEx'*h_i'+meas_std(i)^2;
            % threhold
            thr_i=sqrt(Sigma_t_i)*norminv(1-0.05/(2*n));
        elseif ob_type=="PGO"
            % extend t_i to non-Gaussian overbound
            H_rec = [H_sub(1:i-1,:); h_i; H_sub(i:end,:)];
            P_rec = H_rec*S_subEx;
            IsP_rec = eye(n)-P_rec;
            Isp_rec = IsP_rec(i,:); % i-th row of IsP_rec
            % set definition domain of the t_i
            x_scale=-10:0.01:10;
            scale_list=Isp_rec';
            PHMI=0.05/n;
            % solve distribution and critical value of t_i
            [PL_fake,~,fft_time_all]=YanFuncLib_Overbound_tmp.cal_PL_ex(scale_list,x_scale,[],pgo_current_cells,PHMI);
            % threhold
            thr_i=abs(PL_fake);
        end
        % detection
        if abs(t_i)>thr_i
            is_detect=true;
        end
    end  
end

function is_detect=ss_detector(GTusr_xyz,meas,meas_std,m_sv_pos,s_sv_pos,ref_xyz)
    is_detect=false;
    init_state=zeros(3,1);
    n=length(meas);
    Sigma=diag(meas_std.^2);
    W=inv(Sigma);
    % all-in-view solution
    [eWLSSolution,prn_res,ErrorECEF,H,eDeltaPos,eDeltaPr] = WeightedLeastSquareDD(GTusr_xyz, init_state, meas, meas_std, m_sv_pos,s_sv_pos,ref_xyz);
    lin_pos = eWLSSolution-eDeltaPos;
    S=inv(H'*W*H)*H'*W;
    s3=S(3,:);% z-related
    % subsolution
    for i=1:n
        meas_sub=meas; meas_sub(i,:)=[];
        meas_std_sub=meas_std; meas_std_sub(i,:)=[];
        m_sv_pos_sub=m_sv_pos; m_sv_pos_sub(i,:)=[];
        s_sv_pos_sub=s_sv_pos; s_sv_pos_sub(i,:)=[];
        [eWLSSolution_sub,prn_res_sub,ErrorECEF3d_sub,H_sub,eDeltaPos_sub,eDeltaPr_sub]=WeightedLeastSquareDD(GTusr_xyz, init_state, meas_sub, meas_std_sub, m_sv_pos_sub,s_sv_pos_sub,ref_xyz);
        lin_pos_sub = eWLSSolution_sub-eDeltaPos_sub;
        % test statistic
        d_i=eWLSSolution(3)-eWLSSolution_sub(3);% z-related
        % null distribution of d_i
        Sigma_sub=diag(meas_std_sub.^2);
        W_sub=inv(Sigma_sub);
        S_sub=inv(H_sub'*W_sub*H_sub)*H_sub'*W_sub;
        S_subEx=[S_sub(:,1:i-1), zeros(3,1), S_sub(:,i:end)];
        s3_i=S_subEx(3,:); % z-related
        Sigma_d_i = (s3-s3_i)*Sigma*(s3-s3_i)';
        % threhold
        thr_i=sqrt(Sigma_d_i)*norminv(1-0.05/(2*n));
        % detection
        if abs(d_i)>thr_i
            is_detect=true;
        end
    end  
end

function is_detect=ss_detector_SPP(GTusr_xyz,meas,meas_std,sv_pos)
    is_detect=false;
    init_state=zeros(4,1);
    n=length(meas);
    Sigma=diag(meas_std.^2);
    W=inv(Sigma);
    % all-in-view solution
    [eWLSSolution,prn_res,ErrorECEF,H,eDeltaPos,eDeltaPr] = WeightedLeastSquare(GTusr_xyz, init_state, meas, meas_std, sv_pos);
    lin_pos = eWLSSolution-eDeltaPos;
    S=inv(H'*W*H)*H'*W;
    s3=S(3,:); % z-related
    % subsolution
    for i=1:n
        meas_sub=meas; meas_sub(i,:)=[];
        meas_std_sub=meas_std; meas_std_sub(i,:)=[];
        sv_pos_sub=sv_pos; sv_pos_sub(i,:)=[];
        [eWLSSolution_sub,prn_res_sub,ErrorECEF3d_sub,H_sub,eDeltaPos_sub,eDeltaPr_sub]=WeightedLeastSquare(GTusr_xyz, init_state, meas_sub, meas_std_sub, sv_pos_sub);
        lin_pos_sub = eWLSSolution_sub-eDeltaPos_sub;
        % test statistic
        d_i=eWLSSolution(3)-eWLSSolution_sub(3); % z-related
        % null distribution of d_i
        Sigma_sub=diag(meas_std_sub.^2);
        W_sub=inv(Sigma_sub);
        S_sub=inv(H_sub'*W_sub*H_sub)*H_sub'*W_sub;
        S_subEx=[S_sub(:,1:i-1), zeros(4,1), S_sub(:,i:end)];
        s3_i=S_subEx(3,:); % z-related
        Sigma_d_i = (s3-s3_i)*Sigma*(s3-s3_i)';
        % threhold
        thr_i=sqrt(Sigma_d_i)*norminv(1-0.05/(2*n));
        % detection
        if abs(d_i)>thr_i
            is_detect=true;
        end
    end  
end

function y=GaussPareto_cdf(data)
    y=zeros(size(data));
    for i=1:size(data,2)
        x=data(i);
        y(i) = GaussPareto_cdf_func(x);
    end
end

function [y]=GaussPareto_cdf_func(x)
    global mean_core std_core
    global thr_L xi_L scale_L theta_L
    global thr_R xi_R scale_R theta_R
    
    cdf_thrL=normcdf(thr_L,mean_core,std_core);
    cdf_thrR=normcdf(thr_R,mean_core,std_core);
    gp_L=makedist('gp','k',xi_L,'sigma',scale_L,'theta',theta_L);
    gp_R=makedist('gp','k',xi_R,'sigma',scale_R,'theta',theta_R);
    if x<thr_L
        z=thr_L-x;
        y=cdf_thrL-cdf(gp_L,z)*cdf_thrL;
    elseif x<thr_R
        y=normcdf(x,mean_core,std_core);
    else
        z=x-thr_R;
        y=cdf(gp_R,z)*(1-cdf_thrR)+cdf_thrR;
    end
end

function f = nig_function(x,alpha,beta,delta,mu)
    gama=sqrt(alpha^2-beta^2);
    f = alpha*delta*exp(beta*(x-mu)+delta*gama)*besselk(1,alpha*sqrt(delta^2+(x-mu)^2))/(pi*sqrt(delta^2+(x-mu)^2));
end

function y = nig_pdf(data)
    % This is a simple implementation of the PDF of the NIG distribution.
    % x is the variable, alpha, beta, delta and mu are parameters of the NIG distribution.
    global mu;
    global alpha;
    global delta;
    global beta;pa

    y=zeros(size(data));
    for i=1:size(data,2)
        x=data(i);
        y(i) = nig_function(x,alpha,beta,delta,mu);
    end
end


function dis = Euc_dis(A,B)
    % A, B is row vector
    dis = sqrt((A-B)*(A-B)');
end
