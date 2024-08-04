% clear all
% close all
seed=1234;
addLibPathInit();
YanFuncLib_Overbound_tmp=YanFuncLib_Overbound;
%% explore T1_trans method (20230701)
% % analysis whether T1_trans can be modelled as an overbound
[Xdata,x_lin,pdf_data,cdf_data,gmm_dist]=YanFuncLib_Overbound_tmp.load_GMM(seed);
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
% set folder path and file name
folder = 'C:\Users\Administrator\Desktop\WorkSpace\MyRTKLBAPP\MyRTKLBAPP\rnx2rtkp\.vscode'; 
filePattern = fullfile(folder, '*.csv'); % using wildcard
csvFiles = dir(filePattern); % 
% read each csv
for i = 1:length(csvFiles)
    filename = fullfile(csvFiles(i).folder, csvFiles(i).name);
    data{i} = readtable(filename, 'HeaderLines', 1); % jump first line of header 
end

% merge
bigTable = vertcat(data{:});
% set folder path and file name
filename = 'mergeurbandd.csv'; % new filename
folder = 'C:\Users\Administrator\Desktop\WorkSpace\MyRTKLBAPP\MyRTKLBAPP\rnx2rtkp\.vscode'; % path for saving new file
fullPath = fullfile(folder, filename);

% save to csv
writetable(bigTable, fullPath, 'Delimiter', ',', 'QuoteStrings', true);
% add header manually
% 
% load('Data/mnav_zmp1_jan/mergedRefJan.mat');
% % select: ele(25~50)
% filter_ele=(mergedurbandd.U2I_Elevation>=25 && mergedurbandd.U2I_Elevation<=50); 
% Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele);

%% 20230802 Bias in Principal Gaussian overbound
% seed=1234;
% % load GMM
% % [Xemp,x_lin_emp,pdf_emp,cdf_emp,gmm_dist]=YanFuncLib_Overbound_tmp.load_GMM_bias(seed);
% % [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD({'Data/mnav_zmp1_jan/mergedRefJan.mat'},...
% %     30,5,'2020-01-01','2020-01-31 23:59:59',10,'substract median');% data has human error
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
%                         30,5,inf,40,'substract median');% data has human error
% % Xdata=Xdata-median(Xdata); % should not do this
% % [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
% %                         30,5,inf,40,'substract median');  % data has human error
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


%% 20240322 optimization based gmm fitting
% load('Data/mnav_zmp1_halfyear_20240322/mergedRefhalfyear.mat');
% mergedRefAll=mergedRefhalfyear;
% filter_date = mergedRefAll.datetime>'2020-01-01' & mergedRefAll.datetime<'2020-02-01';
% mergedRefAll = mergedRefAll(filter_date,:);
% item=1;
% figure;
% for ele = 15:5:80
%     filter_ele=(mergedRefAll.U2I_Elevation>=ele & mergedRefAll.U2I_Elevation<=ele+5); 
%     filter_err=(mergedRefAll.doubledifferenced_pseudorange_error>=-10 & mergedRefAll.doubledifferenced_pseudorange_error<=10); 
%     Xdata=mergedRefAll.doubledifferenced_pseudorange_error(filter_ele & filter_err);
%     num=length(Xdata);
%     if num<2500
%         continue
%     end
%     [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
%     NSamples=length(Xdata);
%     lim=max(-min(Xdata),max(Xdata));
%     x_lin = linspace(-lim, lim, NSamples);
%     Y=Xdata';
% 
%     global data ecdf_sim
%     data=sort(Y);
%     ecdf_sim = 1/length(data):1/length(data):1;
% 
%     p1 = optimvar('p1');
%     std1 = optimvar('std1');
%     std2 = optimvar('std2');
% 
%     param0.p1=0.5;
%     param0.std1=std(data);
%     param0.std2=std(data)*2;
% 
%     obj_left = fcn2optimexpr(@objfunx_left,p1,std1,std2);
%     prob_left = optimproblem('Objective',obj_left);
%     prob_left.Constraints.constr1 = p1 <=1-0.05;
%     prob_left.Constraints.constr2 = p1 >=0.5;
%     prob_left.Constraints.constr3 = std1 >=std(Xdata)/2;
%     prob_left.Constraints.constr4 = std1 <=std(Xdata);
%     prob_left.Constraints.constr5 = std2 >=std(Xdata);
%     prob_left.Constraints.constr6 = std2 <=40;
%     prob_left.Constraints.constr7 = std1 <= std2;
% 
%     showproblem(prob_left)
% 
%     [sol_left,fval_left] = solve(prob_left,param0);
% 
% 
%     obj_right = fcn2optimexpr(@objfunx_right,p1,std1,std2);
%     prob_right = optimproblem('Objective',obj_right);
%     prob_right.Constraints.constr1 = p1 <=1-0.05;
%     prob_right.Constraints.constr2 = p1 >=0.5;
%     prob_right.Constraints.constr3 = std1 >=0.00001;
%     prob_right.Constraints.constr4 = std1 <=std(Xdata);
%     prob_right.Constraints.constr5 = std2 >=std(Xdata);
%     prob_right.Constraints.constr6 = std2 <=40;
%     prob_right.Constraints.constr7 = std1 <= std2;
% 
%     showproblem(prob_right)
% 
%     [sol_right,fval_right] = solve(prob_right,param0);
% 
% 
%     sol_new.std1 = max(sol_left.std1,sol_right.std1);
%     sol_new.std2 = max(sol_left.std2,sol_right.std2);
%     if objfunx_left(sol_left.p1,sol_new.std1,sol_new.std2) < objfunx_left(sol_right.p1,sol_new.std1,sol_new.std2)
%       sol_new.p1=sol_left.p1;
%     else
%       sol_new.p1=sol_right.p1;
%     end
%     sol=sol_new;
% 
%     gmm_dist = gmdistribution([0;0], cat(3, sol.std1^2 ,sol.std2^2), [sol.p1,1-sol.p1]);
%     rng(seed);
%     Xgmm_data = random(gmm_dist, 10000);
%     kurtosis(Xgmm_data);
%     
%     cdf_emp=cdf(gmm_dist,x_lin')';
%     %  PGO
%     alpha_adjust=max(0.5,abs(sol.std1-sol.std2)/sol.std2);
% %     alpha_adjust=1/kurtosis(Xdata);
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
%     % Two step Gaussian - use symmetric twp-step bound with defaut param
%     [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin);
%     % Gaussian Pareto
%     [params_gpo,pdf_gpo,cdf_gpo]=YanFuncLib_Overbound_tmp.Gaussian_Pareto_bound(Xdata,x_lin);
% 
%    %% Plot Results 
%     subplot(3,5,item)
%     % ecdf plot
%     h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',0.5,'MarkerSize', 2);
%     hold on
%     % Two step Gaussian plot
%     h21=semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'g','LineWidth',1);
%     % Gaussian Pareto plot
%     h3=semilogy(x_lin,cdf_gpo,'r','LineWidth',1);
%     % GMM plot
%     h4=semilogy(x_lin,cdf_emp,'m-','LineWidth',1);
%     % PGO plot
%     h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',0.5,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%     yline(0.5);
%     xlim([min(x_lin)*1.2,max(x_lin)*0.5]);
%     ylim([1e-6,1]);
%     xlabel('Error (m)');
%     ylabel('CDF (log scale)');
% %     set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% %     A = legend([h1,h21,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','GMM fitting','Principal Gaussian');
% %     set(A,'FontSize',13.5)
%     grid on
%     title(['ele: ',num2str(ele),'   alpha: ',num2str(alpha_adjust)]);
%     item=item+1;
% end
%     
% function f = objfunx_left(p1,std1,std2)
%     global data
%     global ecdf_sim
%     global LeftPortion Lbig_range
%     predict = p1*normcdf(data,0,std1)+(1-p1)*normcdf(data,0,std2);
%     num = length(data);
%     weights = [ecdf_sim(1:floor(num/2)),1+1/num-ecdf_sim(floor(num/2)+1:end)];
%     weights = 1./weights;
%     f = sum(weights.*(abs(predict-ecdf_sim)./ecdf_sim)).^2;
% end
% 
% function f = objfunx_right(p1,std1,std2)
%     global data
%     global ecdf_sim
%     global LeftPortion Lbig_range
%     predict = p1*normcdf(data,0,std1)+(1-p1)*normcdf(data,0,std2);
%     num = length(data);
%     weights = [ecdf_sim(1:floor(num/2)),1+1/num-ecdf_sim(floor(num/2)+1:end)];
%     weights = 1./weights;
%     f = sum(weights.*(abs(predict-ecdf_sim)./(1-ecdf_sim+0.001))).^2;
% end

%% 20240323 explore sigma inflation
% % load('Data/mnav_zmp1_halfyear_20240322/mergedRefhalfyear.mat');
% % load('Data/mnav_zmp1_halfyear_2nd_20240325/mergedRefhalfyear2nd.mat');
% % tmp = vertcat(mergedRefhalfyear, mergedRefhalfyear2nd);
% % mergedRefAll = tmp;
% filter_date = tmp.datetime>'2020-01-01' & tmp.datetime<'2020-02-01';
% mergedRefAll = tmp(filter_date,:);
% mergedRefAll.doubledifferenced_pseudorange_error = -mergedRefAll.doubledifferenced_pseudorange_error; % correct yihan's mistake
% item=1;
% figure;
% for ele = 60:5:90
%     filter_ele=(mergedRefAll.U2I_Elevation>=ele & mergedRefAll.U2I_Elevation<=ele+5); 
%     filter_err=(mergedRefAll.doubledifferenced_pseudorange_error>=-10 & mergedRefAll.doubledifferenced_pseudorange_error<=10); 
%     Xdata=mergedRefAll.doubledifferenced_pseudorange_error(filter_ele & filter_err);
%     % shift the distribution to obtian zero-median
%     Xdata = Xdata - median(Xdata);
% %     Xdata = unique(Xdata);
%     num=length(Xdata);
%     if num<2500*2
%         continue
%     end
%         
%     lim=max(-min(Xdata),max(Xdata));
%     x_lin = linspace(-lim, lim, num);
% %     gmm_dist=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
%     [sol,gmm_dist] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
%     rng(seed);
%     Xgmm_data = random(gmm_dist, 10000);
%     kurtosis(Xgmm_data);
%     
%     % ecdf
% %     [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
%     x_lin_ecdf=sort(Xdata);
%     ecdf_data = linspace(1/length(x_lin_ecdf), 1-1/length(x_lin_ecdf), length(x_lin_ecdf))'; % a better way
%     % gmm emp
%     cdf_emp=cdf(gmm_dist,x_lin')';
%     % PGO
%     alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xdata,gmm_dist);
%     alpha_adjust = min(0.5,alpha_adjust);
% %     min_avoid_cases  = 1e10;
% %     for alpha = 0.05:0.05:1-0.05   
% %         [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha);
% %         predict_cdf = YanFuncLib_Overbound_tmp.two_piece_cdf(x_lin_ecdf',...
% %                             gmm_tmp,params_pgo.xL2p,params_pgo.xR2p);
% %         left_punish_idx = x_lin_ecdf'<0 & predict_cdf<ecdf_data';
% %         right_punish_idx = x_lin_ecdf'>0& predict_cdf>ecdf_data';
% %         avoid_cases = sum(left_punish_idx)+sum(right_punish_idx)
% %         if avoid_cases<min_avoid_cases
% %             min_avoid_cases = avoid_cases;
% %             alpha_adjust = alpha;
% %         end
% %     end
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
%     
%     % check and inflation
%     gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata);
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
%     % gmm emp
%     cdf_gmm_inflate_pgo=cdf(gmm_inflate_pgo,x_lin')';
% 
% %     % check and inflation
% %     gmm_tmp=gmm_dist;
% %     check_complete = false;
% %     while ~check_complete
% %         check_complete = true;
% %         predict_cdf = YanFuncLib_Overbound_tmp.two_piece_cdf(x_lin_ecdf',...
% %                             gmm_tmp,params_pgo.xL2p,params_pgo.xR2p);
% %         % core check
% %         left_punish_idx = x_lin_ecdf'> params_pgo.xL2p & x_lin_ecdf'<0 & predict_cdf<ecdf_data';
% %         right_punish_idx = x_lin_ecdf'< params_pgo.xR2p & x_lin_ecdf'>0 & predict_cdf>ecdf_data';
% %         avoid_cases_core = sum(left_punish_idx)+sum(right_punish_idx)
% %         if avoid_cases_core > 0.05*sum(x_lin_ecdf'> params_pgo.xL2p & x_lin_ecdf'< params_pgo.xR2p) % 5% overbound
% %             gmm_tmp=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_tmp,(1+0.01)^2,1); % 5% inflation on delta_1
% %             gmm_tmp=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_tmp,1,(1+0.01)^2); % 5% inflation on delta_2
% %             min_avoid_cases = 1e10;
% %             for alpha = 0.05:0.05:1-0.05  
% %                 [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_tmp,alpha);
% %                 predict_cdf = YanFuncLib_Overbound_tmp.two_piece_cdf(x_lin_ecdf',...
% %                                     gmm_tmp,params_pgo.xL2p,params_pgo.xR2p);
% %                 left_punish_idx = x_lin_ecdf'> params_pgo.xL2p & x_lin_ecdf'<0 & predict_cdf<ecdf_data';
% %                 right_punish_idx = x_lin_ecdf'< params_pgo.xR2p & x_lin_ecdf'>0 & predict_cdf>ecdf_data';
% %                 avoid_cases = sum(left_punish_idx)+sum(right_punish_idx)
% %                 if avoid_cases<min_avoid_cases
% %                     min_avoid_cases = avoid_cases;
% %                     alpha_adjust = alpha;
% %                 end
% %             end
% %             [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_tmp,alpha_adjust);
% %             predict_cdf = YanFuncLib_Overbound_tmp.two_piece_cdf(x_lin_ecdf',...
% %                                     gmm_tmp,params_pgo.xL2p,params_pgo.xR2p);
% %             check_complete = false;
% %         end
% %         
% %         % tail check
% %         left_punish_idx = x_lin_ecdf'<params_pgo.xL2p & predict_cdf<ecdf_data';
% %         right_punish_idx = x_lin_ecdf'> params_pgo.xR2p& predict_cdf>ecdf_data';
% %         avoid_cases_tail = sum(left_punish_idx)+sum(right_punish_idx)
% %         if avoid_cases_tail>0
% %             gmm_tmp=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_tmp,(1+0.01)^2,1); % 5% inflation on delta_1
% %             gmm_tmp=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_tmp,1,(1+0.01)^2); % 5% inflation on delta_2
% %             min_avoid_cases = 1e10;
% %             for alpha = 0.05:0.05:1-0.05   
% %                 [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_tmp,alpha);
% %                 predict_cdf = YanFuncLib_Overbound_tmp.two_piece_cdf(x_lin_ecdf',...
% %                                     gmm_tmp,params_pgo.xL2p,params_pgo.xR2p);
% %                 left_punish_idx = x_lin_ecdf'<params_pgo.xL2p & predict_cdf<ecdf_data';
% %                 right_punish_idx = x_lin_ecdf'> params_pgo.xR2p& predict_cdf>ecdf_data';
% %                 avoid_cases = sum(left_punish_idx)+sum(right_punish_idx)
% %                 if avoid_cases<min_avoid_cases
% %                     min_avoid_cases = avoid_cases;
% %                     alpha_adjust = alpha;
% %                 end
% %             end
% %             [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_tmp,alpha_adjust);
% %             predict_cdf = YanFuncLib_Overbound_tmp.two_piece_cdf(x_lin_ecdf',...
% %                                     gmm_tmp,params_pgo.xL2p,params_pgo.xR2p);
% %             check_complete = false;
% %         end
% %     end
%     
%     % check and inflation (GMM version)
%     gmm_inflate_pure=gmm_dist;
%     check_complete = false;
%     while ~check_complete
%         check_complete = true;
%         predict_cdf = cdf(gmm_inflate_pure,x_lin_ecdf)';
%         left_punish_idx = x_lin_ecdf'<params_pgo.xL2p & predict_cdf<ecdf_data';
%         right_punish_idx = x_lin_ecdf'> params_pgo.xR2p& predict_cdf>ecdf_data';
%         avoid_cases_tail = sum(left_punish_idx)+sum(right_punish_idx)
%         left_punish_idx = x_lin_ecdf'> params_pgo.xL2p & x_lin_ecdf'<0 & predict_cdf<ecdf_data';
%         right_punish_idx = x_lin_ecdf'< params_pgo.xR2p & x_lin_ecdf'>0 & predict_cdf>ecdf_data';
%         avoid_cases_core = sum(left_punish_idx)+sum(right_punish_idx)
%         
%         if avoid_cases_tail > 0
%             gmm_inflate_pure=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_inflate_pure,(1+0.01)^2,1); % 5% inflation on delta_1
%             gmm_inflate_pure=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_inflate_pure,1,(1+0.01)^2); % 5% inflation on delta_2
%             check_complete = false;
%         end
%         
%         if avoid_cases_core > 0.05*sum(x_lin_ecdf'> params_pgo.xL2p & x_lin_ecdf'< params_pgo.xR2p) % 5% overbound
%             gmm_inflate_pure=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_inflate_pure,(1+0.01)^2,1); % 5% inflation on delta_1
%             gmm_inflate_pure=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_inflate_pure,1,(1+0.01)^2); % 5% inflation on delta_2
%             check_complete = false;
%         end
%     end
%     % gmm emp
%     cdf_gmm_inflate_pure=cdf(gmm_inflate_pure,x_lin')';
%     
%     % Plot Results 
%     subplot(4,3,item)
%     % ecdf plot
%     h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1.5,'MarkerSize', 2);
%     hold on
%     % GMM plot
%     h4=semilogy(x_lin,cdf_emp,'m-','LineWidth',1);
%     % PGO plot
%     h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',0.5,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%     % GMM inflate plote
% %     h6 = semilogy(x_lin,cdf_gmm_inflate_pure,'r-','LineWidth',1);
%     h7 = semilogy(x_lin,cdf_gmm_inflate_pgo,'m:','LineWidth',1);
%     yline(0.5);
%     xlim([min(x_lin)*1.2,max(x_lin)*0.5]);
%     ylim([1e-6,1]);
%     xlabel('Error (m)');
%     ylabel('CDF (log scale)');
%     grid on
%     title(['ele: ',num2str(ele),'   alpha: ',num2str(params_pgo.alpha)]);
%     item=item+1;
%     
%     % Plot Results 
%     subplot(4,3,item)
%     % ecdf plot
%     h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1.5,'MarkerSize', 2);
%     hold on
%     % GMM plot
%     h4=semilogy(x_lin,1-cdf_emp,'m-','LineWidth',1);
%     % PGO plot
%     h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',0.5,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%     % GMM inflate plote
% %     h6 = semilogy(x_lin,1-cdf_gmm_inflate_pure,'r-','LineWidth',1);
%     h7 = semilogy(x_lin,1-cdf_gmm_inflate_pgo,'m:','LineWidth',1);
%     yline(0.5);
% %     xlim([min(x_lin)*1.2,max(x_lin)*0.5]);
%     ylim([1e-6,1]);
%     xlabel('Error (m)');
%     ylabel('CCDF (log scale)');
%     grid on
%     title(['ele: ',num2str(ele),'   alpha: ',num2str(params_pgo.alpha)]);
%     item=item+1;
%     
%     % Plot Results 
%     subplot(4,3,item)
%     boxplot(Xdata);
%     title(['kurtosis: ',num2str(kurtosis(Xdata))]);
%     item=item+1;
%     if item==13
%         break
%     end
% end

%% 20240327 a reasonable way to partition core and tail region
% p1=0.9;
% p2=1-p1;
% mu1=0;
% mu2=0;
% % sigma1=2.6917^2; % 0.5 
% % sigma2=15.4972^2; % 1
% sigma1=0.5^2; % 0.5 
% sigma2=(0.5*2.5)^2; % 1
% gm = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
% Nsamples=10000;
% Xdata=random(gm, Nsamples);
% lim=4;
% x_lin = linspace(-lim, lim, Nsamples);
% alpha = YanFuncLib_Overbound_tmp.find_alpha(Xdata,gm);
% alpha = min(0.5,alpha);
% % alpha=0.3;
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gm,alpha);
% [mean_tsgo, std_tsgo, ~, ~]=YanFuncLib_Overbound_tmp.two_step_bound_zero(Xdata,x_lin);
% 
% % intersection of s1 and s2
% x_its_left = -sqrt( (2*sigma1*sigma2/(sigma2-sigma1)) ...
%                 *log(p1*sqrt(sigma2)/(p2*sqrt(sigma1))) );
% x_its_right = -x_its_left;         
%             
% % member weight
% figure;
% subplot(1,2,1)
% yyaxis left
% h1=plot(x_lin,pdf(gm,x_lin'),'k','LineWidth',2);
% ylabel('PDF');
% yyaxis right
% h2=plot(x_lin,params_pgo.s1_list,'r','LineWidth',2);
% hold on
% h3=plot(x_lin,params_pgo.s2_list,'b','LineWidth',2);
% scatter(x_its_left,0.5,72,'bo','filled');
% scatter(x_its_right,0.5,72,'ro','filled');
% h4=xline(params_pgo.xL2p,'k--','LineWidth',1.5);
% h5=xline(params_pgo.xR2p,'r--','LineWidth',1.5);
% h6=fill([params_pgo.xL2p params_pgo.xL2p  params_pgo.xR2p params_pgo.xR2p],[0 1.5 1.5 0],'k','FaceAlpha',0.1);
% ylim([0,1.5]);
% ylabel('Membership Weight');
% xlabel('Error (m)');
% ax = gca;
% ax.YAxis(1).Color = 'black';
% ax.YAxis(2).Color = 'black';
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h2,h3,h4,h5,h6],'BGMM','s1(x)','s2(x)','xlp','xrp','Core region');
% set(A,'FontSize',13.5)
% grid on
% 
% subplot(1,2,2)
% % total_Sigma = p1*sigma1+p2*sigma2;
% total_Sigma = sigma1;
% tnorm = makedist('Normal','mu',0,'sigma',sqrt(total_Sigma));
% tnorm_st = makedist('Normal','mu',0,'sigma',1);
% Xdata_tnorm=random(tnorm, Nsamples,1);
% Xdata_tnorm_st = random(tnorm_st, Nsamples,1);
% 
% kur_arr = [];
% kur_tnorm_arr = [];
% kur_tnorm_st_arr=[];
% alpha_arr = [];
% x_div_arr = [];
% for x_div = x_lin(x_lin<0)
%     Xdata_use = Xdata(Xdata>x_div & Xdata<-x_div);
%     truncated_rate = 1-length(Xdata_use)/length(Xdata);
%     if length(Xdata_use)<1000
%         break
%     end
%     
%     kur_arr(end+1)=kurtosis(Xdata_use);
%     
%     Xdata_tnorm_use = Xdata_tnorm(Xdata_tnorm>x_div & Xdata_tnorm<-x_div);
%     kur_tnorm_arr(end+1)=kurtosis(Xdata_tnorm_use);
%     
%     x_st_div = norminv(truncated_rate/2);
%     Xdata_tnorm_st_use = Xdata_tnorm_st(Xdata_tnorm_st>x_st_div & Xdata_tnorm_st<-x_st_div);
%     kur_tnorm_st_arr(end+1)=kurtosis(Xdata_tnorm_st_use);
%     
%     [~,s2]=YanFuncLib_Overbound_tmp.cal_omega(x_div,mu1,sigma1,p1,mu2,sigma2,p2);
%     alpha_arr(end+1) = s2;
%     
%     x_div_arr(end+1) = x_div;
%     
%     if abs(x_its_left-x_div) < abs(x_lin(2)-x_lin(1))
%         % relative kutorsis error at intersection point
%         ist_kur_re = (kur_arr(end)-kur_tnorm_st_arr(end))/kur_tnorm_st_arr(end);
%     end
% end
% 
% h1=plot(x_div_arr,100*(kur_arr-kur_tnorm_st_arr)./kur_tnorm_st_arr,'m','LineWidth',1.5);
% hold on
% % yline(0.05*100,'k:');
% % yline(-0.05*100,'k:');
% % h2=plot(x_div_arr,alpha_arr,'b');
% h3=fill([min(x_div_arr),min(x_div_arr),max(x_div_arr),max(x_div_arr)],[-0.05,0.05,0.05,-0.05]*100,'k','FaceAlpha',0.1);
% % xline(x_its_left,'k--','LineWidth',1.5);
% h4 = scatter(x_its_left,ist_kur_re*100,72,'bo','filled');
% xlim([min(x_div_arr),max(x_div_arr)])
% ylim([-0.2,1.2]*100)
% ylabel('Percent (%)');
% xlabel('Error (m)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h3,h4],'Relative kurtosis error','\pm 5% error','Left intersection');
% set(A,'FontSize',13.5)
% grid on
% 
% % pdf
% figure;
% h1=plot(x_lin,pdf(gm,x_lin'),'k','LineWidth',2);
% hold on
% % PGO
% h2=plot(x_lin,pdf_pgo,'b','LineWidth',2);
% % Two step Gaussian
% h3=plot(x_lin,normpdf(x_lin,mean_tsgo,std_tsgo),'g','LineWidth',2);
% h4=xline(params_pgo.xL2p,'k--','LineWidth',1);
% h5=xline(params_pgo.xR2p,'r--','LineWidth',1);   
% xlabel('Error');
% ylabel('PDF');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h2,h3,h4,h5],'BGMM','Principal Gaussian','Gaussian','xlp','xrp');
% set(A,'FontSize',13.5)
% grid on
% 
% 
% % cdf
% figure;
% h1=plot(x_lin,cdf(gm,x_lin'),'k','LineWidth',2);
% hold on
% % PGO
% h2=plot(x_lin,cdf_pgo,'b','LineWidth',2);
% % Two step Gaussian
% h3=plot(x_lin,normcdf(x_lin,mean_tsgo,std_tsgo),'g','LineWidth',2);
% h4=xline(params_pgo.xL2p,'k--','LineWidth',1);
% h5=xline(params_pgo.xR2p,'r--','LineWidth',1);   
% xlabel('Error');
% ylabel('CDF');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h2,h3,h4,h5],'BGMM','Principal Gaussian','Gaussian','xlp','xrp');
% set(A,'FontSize',13.5)
% grid on

%% 20240403 visualize hist of Urban DGNSS error in each bin
load('Data/urban_dd_0816/mergeurbandd.mat');
figure
filter_SNR=(mergedurbandd.U2I_SNR>=40); 
item=1;
for ele = [30,35,40,45,50,55]
    filter_ele=(mergedurbandd.U2I_Elevation>=ele & mergedurbandd.U2I_Elevation<=ele+5); 
    Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele & filter_SNR);
    subplot(2,3,item)
    histogram(Xdata,'normalization','pdf')
%     xlim([-27,27])
%     ylim([-0.01,0.5])
    str1 = ['Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ'];
    str2 = ['Min: ',num2str(min(Xdata)),' ,Max:',num2str(max(Xdata))];
    str3 = ['Mean: ',num2str(mean(Xdata)),' ,Median:',num2str(median(Xdata)),' ,Amount:',num2str(length(Xdata))];
    str4 = ['Mean: ',num2str(mean(Xdata)),' ,Amount:',num2str(length(Xdata))];
    title({str1,str4});
    item=item+1;
end

%% 20240406 discretization method
% N = 50-1;  % total number of intervals
% half_N = N/2;
% tranc = N/10; % should adjust to achive unimodal
% x_lim = 10;
% leftEdge_list=calLeftEdges(N,x_lim,tranc,'equal');
% % cdf at each left edge
% cdf_leftEdge_list = normcdf(leftEdge_list);
% % overbound cdf
% obcdf_leftEdge_list=[];
% for i=1:N-1
%     if i<half_N
%         obcdf_leftEdge = normcdf(leftEdge_list(i+1));
%     else
%         obcdf_leftEdge = normcdf(leftEdge_list(i));
%     end
%     obcdf_leftEdge_list(end+1)=obcdf_leftEdge;
% end
% % transfer ob_cdf to pmf
% pmf_leftEdge_list = [cdf_leftEdge_list(1),cdf_leftEdge_list(2:end)-cdf_leftEdge_list(1:end-1)];
% 
% figure
% stem(leftEdge_list,[1:half_N,flip(1:half_N-1)]);
% 
% figure
% plot(-x_lim:0.1:x_lim,normcdf(-x_lim:0.1:x_lim));
% hold on 
% for i=1:length(leftEdge_list)-1
%     left_edge = leftEdge_list(i);
%     right_edge = leftEdge_list(i+1);
%     cdf_value = obcdf_leftEdge_list(i);
%     line([left_edge, right_edge], [cdf_value,cdf_value], 'Color', 'r', 'LineWidth', 0.5);
% end
% 
% figure
% plot(-x_lim:0.1:x_lim,normpdf(-x_lim:0.1:x_lim));
% figure
% stem(leftEdge_list,pmf_leftEdge_list)
% 
% x_lim = 5;
% delta_x = 0.1;
% [leftEdge_list,obcdf_leftEdge_list,pmf_leftEdge_list]=cal_ob_pmf(x_lim,delta_x,@normcdf);
% 
% figure
% subplot(2,2,1)
% plot(-x_lim:0.1:x_lim,normcdf(-x_lim:0.1:x_lim));
% hold on 
% for i=1:length(leftEdge_list)-1
%     left_edge = leftEdge_list(i);
%     right_edge = leftEdge_list(i+1);
%     cdf_value = obcdf_leftEdge_list(i);
%     line([left_edge, right_edge], [cdf_value,cdf_value], 'Color', 'r', 'LineWidth', 0.5);
% end
% 
% subplot(2,2,2)
% plot(-x_lim:0.1:x_lim,normpdf(-x_lim:0.1:x_lim));
% subplot(2,2,3)
% stem(leftEdge_list,pmf_leftEdge_list)

%% 20240407 MFloats of FFT on my latpot
% N=256;
% arr=complex(zeros(1,N),zeros(1,N));
% tic;
% cnts=1000000;
% for i=1:cnts
% fft_pdf1=fft(arr,N);
% end
% use_time = toc;
% disp(use_time);
% mflops = 5*N*log2(N)/(use_time/cnts * 10^6);
% disp(mflops)

%% 20240409 EM & discrepency fitting comparison
% % Through this experiment, we finally decide to use EM algorihtm for
% % fitting. Due to the sigma inflation strategy, we do not have harsh
% % reguirements in the fitting performance.
% seed=1234;
% ele = 30;
% % gene GMM Data
% gm = gmdistribution([0; 0], cat(3, 1, 36), [0.7 0.3]);
% Xdata = random(gm, 500);
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
%                         30,5,inf,40,'substract median'); % data has human error
% 
% % ecdf
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% % GMM fit - discrepency
% [sol,gmm_dist_op] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
% cdf_gmm_op=cdf(gmm_dist_op,x_lin_ecdf)';
% % GMM fit - EM
% [gmm_dist_em]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% cdf_gmm_em=cdf(gmm_dist_em,x_lin_ecdf)';
% 
% figure
% % ecdf plot
% h1=semilogy(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6);
% hold on
% % gmm_op plot
% h2=semilogy(x_lin_ecdf,cdf_gmm_op,'r','LineWidth',1);
% % gmm_em plot
% h3=semilogy(x_lin_ecdf,cdf_gmm_em,'bs-','LineWidth',1,'MarkerFaceColor','b','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin_ecdf)/26):length(x_lin_ecdf));
% 
% 
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h2,h3],'Sample dist.','Discrepency-based fitting','EM-based fitting');
% set(A,'FontSize',13.5)
% legend
% grid on

%% 20240411 compare Gaussian, paired overbound, core overbound, Two-step Gaussian
% figure
% Xdata = randn(10000,1);
% subplot(2,2,1)
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% plot(x_lin_ecdf,ecdf_data,'*')
% xlim([-4,4])
% hold on
% x_lin=-5:0.01:5;
% plot(x_lin,normcdf(x_lin,0,2),'r','LineWidth',2)
% title('Gaussian CDF overbound')
% xlabel('Error')
% ylabel('PDF')
% 
% subplot(2,2,2)
% plot(x_lin_ecdf,ecdf_data,'*')
% xlim([-4,4])
% hold on
% x_lin=-5:0.01:5;
% plot(x_lin,normcdf(x_lin,-2,1),'r','LineWidth',2)
% plot(x_lin,normcdf(x_lin,2,1),'r','LineWidth',2)
% title('Paired Gaussian overbound')
% xlabel('Error')
% ylabel('PDF')
% 
% subplot(2,2,3)
% plot(x_lin_ecdf,ecdf_data,'*')
% xlim([-4,4])
% hold on
% x_lin=-2.5:0.01:2.5;
% plot(x_lin,normcdf(x_lin,0,1.5),'r','LineWidth',2)
% left_x_lin=-5:0.01:-2.5;
% plot(left_x_lin,normcdf(-2.5,0,1.5)*ones(size(left_x_lin)),'r','LineWidth',2)
% right_x_lin=2.5:0.01:5;
% plot(right_x_lin,normcdf(2.5,0,1.5)*ones(size(right_x_lin)),'r','LineWidth',2)
% title('Gaussian core overbound')
% xlabel('Error')
% ylabel('PDF')
% 
% subplot(2,2,4)
% plot(x_lin_ecdf,ecdf_data,'*')
% xlim([-4,4])
% hold on
% left_x_lin=-5:0.01:0;
% plot(left_x_lin,normcdf(left_x_lin,-2,1),'r','LineWidth',2)
% right_x_lin=0:0.01:5;
% plot(right_x_lin,normcdf(right_x_lin,2,1),'r','LineWidth',2)
% title('Two-step Gaussian overbound')
% xlabel('Error')
% ylabel('PDF')

%% 20240705 explore 1-hour urban data distribution with time
filter_SNR=(mergedurbandd.U2I_SNR>=40); 
ele = 25
filter_ele=(mergedurbandd.U2I_Elevation>=ele & mergedurbandd.U2I_Elevation<=ele+5);
filter_datetime=(mergedurbandd.datetime>"2024-06-28 06:01:59");
ufilter_datetime=(mergedurbandd.datetime>"2024-06-28 06:55:51" & mergedurbandd.datetime<"2024-06-28 06:56:47" );
tmp=mergedurbandd(filter_datetime & ~ufilter_datetime & filter_ele & filter_SNR,:);
% tmp=mergedurbandd(filter_datetime & filter_ele & filter_SNR,:);
Xdata=tmp.doubledifferenced_pseudorange_error;
sortedTable = sortrows(tmp, 'datetime');
figure
subplot(2,1,1);scatter(sortedTable.datetime,sortedTable.doubledifferenced_pseudorange_error,2,'*'); ylim([-5,5]); yline(0);
title(num2str(ele));
subplot(2,1,2);histogram(Xdata,'normalization','pdf'); xlim([-5,5]);
str3 = ['Mean: ',num2str(mean(Xdata)),' ,Median:',num2str(median(Xdata)),' ,N:',num2str(length(Xdata))];
title(str3);

function [leftEdge_list,obcdf_leftEdge_list,pmf_leftEdge_list]=cal_ob_pmf(x_lim,delta_x,cdf_func)
    leftEdge_list = -x_lim:delta_x:x_lim;
    N=length(leftEdge_list)+1; % total number of intervals
    half_N = N/2;
    % cdf at each left edge
    cdf_leftEdge_list = cdf_func(leftEdge_list);
    % overbound cdf
    obcdf_leftEdge_list=[];
    for i=1:N-1
        if i<half_N
            obcdf_leftEdge = cdf_func(leftEdge_list(i+1));
        elseif i<N-2
            obcdf_leftEdge = cdf_func(leftEdge_list(i));
        else
            obcdf_leftEdge = cdf_leftEdge_list(i);
        end
        obcdf_leftEdge_list(end+1)=obcdf_leftEdge;
    end
    % transfer ob_cdf to pmf
    pmf_leftEdge_list = [obcdf_leftEdge_list(1)-cdf_leftEdge_list(1),obcdf_leftEdge_list(2:end)-obcdf_leftEdge_list(1:end-1)];
%     pmf_leftEdge_list = [cdf_leftEdge_list(1),cdf_leftEdge_list(2:end)-cdf_leftEdge_list(1:end-1)];
end

function leftEdge_list=calLeftEdges(N,x_lim,tranc,method)
    if method == "unequal"
        leftEdge_list=[];
        half_N = N/2;
        C=abs(x_lim/log(1/half_N));
        core_interval=-2*C * log((half_N-tranc)/half_N);
        last_bound=0;
        for k = 1:2*half_N-1
            if k<=half_N-tranc
                Bound = C * log(k/half_N);
            elseif k<=half_N+tranc
                Bound = last_bound+core_interval/(2*tranc);
            else
                Bound =  -C * log((2*half_N-k)/half_N);
            end
            leftEdge_list(end+1)=Bound;
            last_bound = Bound;
        end
    else
        ex_half_N = (N+1)/2;
        left_side = linspace(-x_lim, 0,ex_half_N);
        right_side = linspace(0,x_lim,ex_half_N);
        leftEdge_list = [left_side(1:end-1),right_side(2:end)];
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
