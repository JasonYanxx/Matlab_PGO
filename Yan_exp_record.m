clear all
close all
seed=1234;
YanFun=Yan_functions;
%% explore T1_trans method (20230701)
% % analysis whether T1_trans can be modelled as an overbound
% [Xdata,x_lin,pdf_data,cdf_data,gmm_dist]=YanFun.load_GMM(seed);
% YanFun.compare_twoside_bound(Xdata,x_lin,gmm_dist)

%% explore Fault detection (20230702)
% % explore how conservatism will affect fault detection performance
% % GMM
% [Xdata,x_lin,pdf_data,cdf_data,gmm_dist]=YanFun.load_GMM(seed);
% % Two-step Gaussian overbound (zero-mean)
% [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFun.two_step_bound_zero(Xdata,x_lin);
% % Total Gaussian overbound
% [mean_tgo, std_tgo, pdf_tgo, cdf_tgo]=YanFun.total_Gaussian_bound(Xdata,x_lin,gmm_dist);
% 
% % FDE params
% alpha=0.05;
% num=6;
% bias=quantile(Xdata,0.001/2);
% % different method
% [FA_Gaussian,MD_Gaussian]=YanFun.FDE_Gaussian(alpha,seed,num,gmm_dist,bias,std_tsgo^2);
% [FA_tGaussian,MD_tGaussian]=YanFun.FDE_Gaussian(alpha,seed,num,gmm_dist,bias,std_tgo^2);
% [FA_Bayes_max,MD_Bayes_max]=YanFun.FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,"max");
% [FA_Bayes_sum,MD_Bayes_sum]=YanFun.FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,"sum");
% % Monte-Carlo simulation
% [FA_arr,MD_arr,var_arr]=YanFun.FDE_mc_compare(alpha,seed,num);

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
% X = YanFun.customrand(@nig_pdf, interval, N, M)';
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
% pdf_pb=YanFun.two_piece_pdf(x,gm,-xp,xp);
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
% % pdf_gm_conv2=pdf_gm_conv2/sum(pdf_gm_conv2*dx); % 归一化PDF
% [pdf_gm_conv2,~]=YanFun.get_conv(x,pdf_gm,pdf_gm);
% plot(x,pdf_gm_conv2,'k--','linewidth',1);
% tic;
% fft_pdf2=fft(pdf_pb,2*N-1);
% pdf_recon2 = ifft(fft_pdf2 .* fft_pdf2);
% fft_conv_t=toc;
% pdf_recon2=pdf_recon2/sum(pdf_recon2*dx); % 归一化PDF
% pdf_recon2=pdf_recon2(1,floor(N/2):floor(N/2)+N-1); % cut
% plot(x,abs(pdf_recon2),'b--','linewidth',0.5);
% % tic;
% % pdf_dconv2=conv(pdf_pb,pdf_pb);
% % conv_t=toc;
% % pdf_dconv2=pdf_dconv2/sum(pdf_dconv2*dx); % 归一化PDF
% % pdf_dconv2=pdf_dconv2(1,floor(N/2):floor(N/2)+N-1); %cut
% [pdf_dconv2,conv_t]=YanFun.get_conv(x,pdf_pb,pdf_pb);
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
% [pdf_conv_org,x_fftconv,conv_t_org]=YanFun.distConv_org(x,x,pdf_data,pdf_data,"direct");
% [pdf_fftconv_org,x_fftconv_org,fft_conv_t_org]=YanFun.distConv_org(x,x,pdf_data,pdf_data,"fft");
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
% global thr_R xi_R sigma_R theta_R
% [thr_R,theta_R,xi_R,sigma_R]=YanFun.gp_tail_overbound(Xdata);
% % left-tail overbound
% global thr_L xi_L sigma_L theta_L
% [thr_L,theta_L,xi_L,sigma_L]=YanFun.gp_tail_overbound(-Xdata);
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
% % 设置文件夹路径和文件名
% folder = 'Data/urban_dd_0816'; % 文件夹路径
% filePattern = fullfile(folder, '*.csv'); % 文件名通配符
% csvFiles = dir(filePattern); % 匹配文件夹中所有符合通配符的 CSV 文件
% 
% % 循环读取每个 CSV 文件
% for i = 1:length(csvFiles)
%     filename = fullfile(csvFiles(i).folder, csvFiles(i).name);
%     data{i} = readtable(filename, 'HeaderLines', 1); % 跳过表头的第一行
% end
% 
% % 合并所有数据表格为一个大数据表格
% bigTable = vertcat(data{:});
% % 设置文件名和文件路径
% filename = 'merged_urban_dd.csv'; % 新文件名
% folder = 'Data/urban_dd_0816'; % 新文件保存路径
% fullPath = fullfile(folder, filename);
% 
% % 保存表格为 CSV 文件
% writetable(bigTable, fullPath, 'Delimiter', ',', 'QuoteStrings', true);
% % 手动加表头
% 
% load('Data/mnav_zmp1_jan/mergedRefJan.mat');
% % select: ele(25~50)
% filter_ele=(mergedurbandd.U2I_Elevation>=25 && mergedurbandd.U2I_Elevation<=50); 
% Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele);

%% 20230802 Bias in Principal Gaussian overbound
% YanFun=Yan_functions;
% seed=1234;
% % load GMM
% % [Xemp,x_lin_emp,pdf_emp,cdf_emp,gmm_dist]=YanFun.load_GMM_bias(seed);
% % [Xdata,x_lin,pdf_data]=YanFun.load_RefDD();
% [Xdata,x_lin,pdf_data]=YanFun.load_UrbanDD();
% % Xdata=Xdata-median(Xdata); % should not do this
% % [Xdata,x_lin,pdf_data]=YanFun.load_UrbanDD();
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
% gmm_dist_raw=YanFun.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFun.inflate_GMM(gmm_dist_raw,1,2.2) % inflate_ref30: (1,1.15); inflate_urban_30-80: (1.2,2); inflate_urban_30:
% [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,0.9); %ref30:0.9; ref60:0.7; urban_30-80:0.9
% [s1_list,s2_list]=YanFun.gen_s1_s2(x_lin,Xdata,gmm_dist,0,ax);
% plot(ax,x_lin,pdf_pgo,'g','LineWidth',2);
% 
% ax=subplot(1,3,2);
% % Principal Gaussian overbound (left)
% Xleft=Xdata(Xdata<Xmedian);
% Xleft_recon=[Xleft;2*Xmedian-Xleft;Xmedian];
% gmm_dist_left=YanFun.gene_GMM_EM_zeroMean(Xleft_recon-mean(Xleft_recon));
% gmm_dist_left=YanFun.inflate_GMM(gmm_dist_left,1,1.5) % inflate: 1.1; inflate: 1.5
% [s1_list,s2_list]=YanFun.gen_s1_s2(x_lin,Xleft_recon,gmm_dist_left,mean(Xleft_recon),ax);
% [params_pgo_left, pdf_pgo_left, cdf_pgo_left]=YanFun.Principal_Gaussian_bound(Xleft_recon-mean(Xleft_recon),x_lin,gmm_dist_left,0.7);
% plot(ax,x_lin+mean(Xleft_recon),pdf_pgo_left,'g','LineWidth',2);
% 
% ax=subplot(1,3,3);
% % Principal Gaussian overbound (right)
% Xright=Xdata(Xdata>Xmedian);
% Xright_recon=[Xright;2*Xmedian-Xright;Xmedian];
% gmm_dist_right=YanFun.gene_GMM_EM_zeroMean(Xright_recon-mean(Xright_recon));
% gmm_dist_right=YanFun.inflate_GMM(gmm_dist_right,1,1.5) % inflate: 1.1;inflate: 1.5
% [s1_list,s2_list]=YanFun.gen_s1_s2(x_lin,Xright_recon,gmm_dist_right,mean(Xright_recon),ax);
% [params_pgo_right, pdf_pgo_right, cdf_pgo_right]=YanFun.Principal_Gaussian_bound(Xright_recon-mean(Xright_recon),x_lin,gmm_dist_right,0.9);
% plot(ax,x_lin+mean(Xright_recon),pdf_pgo_right,'g','LineWidth',2);
% 
% % cdf
% figure;
% h1=plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
% hold on
% % Two step Gaussian
% [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFun.two_step_bound(Xdata,x_lin);
% h21=plot(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'r','LineWidth',2);
% h22=plot(x_lin(params.idx+1:end),cdf_left_tsgo(params.idx+1:end),'y','LineWidth',1);
% h23=plot(x_lin(1:params.idx),cdf_right_tsgo(1:params.idx),'y--','LineWidth',1);
% h24=plot(x_lin(params.idx+1:end),cdf_right_tsgo(params.idx+1:end),'r--','LineWidth',2);
% % Gaussian Pareto
% [params_gpo,pdf_gpo,cdf_gpo]=YanFun.Gaussian_Pareto_bound(Xdata,x_lin);
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
% x_lin2=linspace(-3, 3, 10000); % solve: 错误使用 griddedInterpolant
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

function y=GaussPareto_cdf(data)
    y=zeros(size(data));
    for i=1:size(data,2)
        x=data(i);
        y(i) = GaussPareto_cdf_func(x);
    end
end

function [y]=GaussPareto_cdf_func(x)
    global mean_core std_core
    global thr_L xi_L sigma_L theta_L
    global thr_R xi_R sigma_R theta_R
    
    cdf_thrL=normcdf(thr_L,mean_core,std_core);
    cdf_thrR=normcdf(thr_R,mean_core,std_core);
    gp_L=makedist('gp','k',xi_L,'sigma',sigma_L,'theta',theta_L);
    gp_R=makedist('gp','k',xi_R,'sigma',sigma_R,'theta',theta_R);
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
    global beta;

    y=zeros(size(data));
    for i=1:size(data,2)
        x=data(i);
        y(i) = nig_function(x,alpha,beta,delta,mu);
    end
end
