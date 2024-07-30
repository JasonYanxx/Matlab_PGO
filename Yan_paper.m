addLibPathInit();
YanFuncLib_Overbound_tmp=YanFuncLib_Overbound;
%% Paper-demo: Fig. 1 & 2
% close all
% p1=0.9;
% p2=1-p1;
% mu1=0;
% mu2=0;
% sigma1=0.5^2; % 0.5 
% sigma2=1^2; % 1
% gm = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
% Nsamples=10001;
% Xdata=random(gm, Nsamples);
% lim=max(-min(Xdata),max(Xdata));
% x_lin = linspace(-lim, lim, Nsamples);
% alpha = 0.7;
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gm,alpha);
% [mean_tsgo, std_tsgo, ~, ~]=YanFuncLib_Overbound_tmp.two_step_bound_zero(Xdata,x_lin);
% 
% % member weight
% figure;
% yyaxis left
% h1=plot(x_lin,pdf(gm,x_lin'),'k','LineWidth',2);
% ylabel('PDF');
% yyaxis right
% h2=plot(x_lin,params_pgo.s1_list,'r','LineWidth',2);
% hold on
% h3=plot(x_lin,params_pgo.s2_list,'b','LineWidth',2);
% h4=xline(params_pgo.xL2p,'k--','LineWidth',1.5);
% h5=xline(params_pgo.xR2p,'r--','LineWidth',1.5);
% ylim([0,1.5]);
% ylabel('Membership Weight');
% xlabel('Error');
% ax = gca;
% ax.YAxis(1).Color = 'black';
% ax.YAxis(2).Color = 'black';
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h2,h3,h4,h5],'BGMM','s1(x)','s2(x)','xlp','xrp');
% set(A,'FontSize',13.5)
% grid on
% 
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

%% Paper-Overbounding compare: Fig.4,5,6,9
figure
seed=1234;
use_subplots=false;
ele = 30;
void_ratio = 1/100;% 0.1% for urban; 1% for ref


% load Data
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD(...
%     {'Data/mnav_zmp1_halfyear_20240410/mergedRefhalfyear.mat','Data/mnav_zmp1_halfyear_2nd_20240410/mergedRefhalfyear2nd.mat'},...
%     ele,5,'2020-01-01 00:00:00','2020-01-31 23:59:59',10,'mirror data','substract median');
[Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD({'Data/mnav_zmp1_jan_20240409/mergedRefJan.mat'},...
    ele,5,'2020-01-01','2020-01-31 23:59:59',10,'mirror data','substract median');
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
%                         30,5,inf,40,'substract median'); % TAES draft (data has human error)
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat'},...
%                         ele,5,inf,35,'substract median','TAES_2nd_complementary');
% str_title =['1 year of DGNSS Errors (Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ)'];
str_title =['Urban DGNSS Errors (Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ)'];

% ecdf
[ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% x_lin_ecdf=sort(Xdata);
% ecdf_data = linspace(1/length(x_lin_ecdf), 1-1/length(x_lin_ecdf), length(x_lin_ecdf))'; % a better way

% gmm fit as emperical
%[sol,gmm_dist] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
[gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
rng(seed);
Xgmm_data = random(gmm_dist, 10000);
kurtosis(Xgmm_data);
pdf_emp=pdf(gmm_dist,x_lin')';
cdf_emp=cdf(gmm_dist,x_lin')';

% Principal Gaussian overbound (zero-mean)
% alpha_adjust =  0;
% lpha_adjust =  exp(3-kurtosis(Xdata));
% alpha_adjust =  (3/kurtosis(Xdata));
% alpha_adjust = 3/kurtosis(Xgmm_data)*0.5;
% alpha_adjust=0.15;
% alpha_adjust1=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% alpha_adjust2=YanFuncLib_Overbound_tmp.find_alpha(-Xgmm_data,gmm_dist);
% alpha_adjust = min([0.5,alpha_adjust1,alpha_adjust2]);
alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
alpha_adjust = min(0.5,alpha_adjust);
[params_pgo, pdf_pgo, cdf_pgo_pre]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
% check and inflation
gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
[params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
% gmm emp
cdf_gmm_inflate_pgo=cdf(gmm_inflate_pgo,x_lin')';

% Two step Gaussian
% fix bug: 20240319 - use symmetric twp-step bound with defaut param
[params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin);

% Gaussian Pareto
[params_gpo,pdf_gpo,cdf_gpo]=YanFuncLib_Overbound_tmp.Gaussian_Pareto_bound(Xdata,x_lin);

% qq plot
if ~use_subplots
    figure
else
    subplot(2,2,1)
end
Xnorm=randn(1,length(Xdata));
h=qqplot(Xdata,Xnorm);
xlabel('Quantiles of error distribution (m)');
ylabel('Standard normal quantile (m)');
title(str_title);
set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
grid on

% cdf plot
if ~use_subplots
    figure
else
    subplot(2,2,2)
end
% ecdf plot
h1=plot(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6,'MarkerIndices',1:floor(length(x_lin_ecdf)/30):length(x_lin_ecdf));
hold on
% Two step Gaussian plot
h21=plot(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'go-','LineWidth',1.5,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
h24=plot(x_lin(params.idx+1:end),cdf_right_tsgo(params.idx+1:end),'g--','LineWidth',1.5);
% Gaussian Pareto plot
h3=plot(x_lin,cdf_gpo,'r','LineWidth',2);
% GMM plot
h4=plot(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% PGO plot
h5=plot(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/24):length(x_lin));
xlabel('Error (m)');
ylabel('CDF');
title(str_title);
set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
A = legend([h1,h21,h24,h3,h4,h5],'Sample dist.','Two-step Gaussian (L)','Two-step Gaussian (R)','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SE');
set(A,'FontSize',13.5)
grid on

% log scale cdf plot (left side) 
if ~use_subplots
    figure
else
    subplot(2,2,3)
end
% ecdf plot
h1=semilogy(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6);
hold on
% Two step Gaussian plot
h21=semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'go-','LineWidth',1.5,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% Gaussian Pareto plot
h3=semilogy(x_lin,cdf_gpo,'r','LineWidth',2);
% GMM plot
h4=semilogy(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% PGO plot
h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% yline(0.5);
xlim([min(x_lin)*1.2,max(x_lin)*0.5]);
ylim([1e-7,1]);
xlabel('Error (m)');
ylabel('CDF (log scale)');
title(str_title);
set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
A = legend([h1,h21,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SE');
set(A,'FontSize',13.5)
grid on

% log scale cdf plot (right side)
if ~use_subplots
    figure
else
    subplot(2,2,4)
end
% ecdf plot
h1=semilogy(x_lin_ecdf,1-ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6);
hold on
% Two step Gaussian plot
h24=semilogy(x_lin(params.idx+1:end),1-cdf_right_tsgo(params.idx+1:end),'go-','LineWidth',1.5,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% Gaussian Pareto plot
h3=semilogy(x_lin,1-cdf_gpo,'r','LineWidth',2);
% GMM plot
h4=semilogy(x_lin,1-cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% PGO plot
h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% yline(0.5);
xlim([min(x_lin)*0.5,max(x_lin)*1.2]);
ylim([1e-7,1]);
xlabel('Error (m)');
ylabel('CCDF (log scale)');
title(str_title);
set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
A = legend([h1,h24,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SW');
set(A,'FontSize',13.5)
grid on

%% Paper-Urban DGNSS error against SNR and Ele: Fig.8
% load('Data/urban_dd_0816/mergeurbandd.mat'); % TAES draft (data is wrong)
% % load('Data/urban_dd_20240104/mergeurbandd.mat'); % data is corrected
% % load('Data/mnav_zmp1_jan/mergedRefJan.mat');
% figure;
% % set color map
% cmap = hot; % use jet color map
% cmap = flipud(cmap);
% c = abs(mergedurbandd.doubledifferenced_pseudorange_error); % z-axis as color
% c=log(c);
% % plot 2D scatter; use z-axis as color
% scatter(mergedurbandd.U2I_Elevation, mergedurbandd.U2I_SNR, 30, c, 'filled');
% colormap(cmap);
% caxis([-1, 6]);
% colorbar;
% xlabel('Elevation angle (degree)');
% ylabel('Signal to noise ratio (dB)');
% title('Urban DGNSS Errors')
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');

%% VPL and VPE series, PL compuation time: Fig. 7a,b £¨TAES draft£©
% seed=1234;
% % load GMM
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
%                         30,5,inf,40,'substract median'); % data has human error
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% counts=length(x_lin_org);
% % Principal Gaussian overbound (zero-mean)
% gmm_dist_raw=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_raw,2,1.5); % inflate: 1.15; inflate: (2,1.5)
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin_org,gmm_dist,0.7);
% % two-step Gaussian overbound (zero-mean)
% [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_zero(Xdata,x_lin_org);
% % expand the definition domian of the range domain error
% lim=50;
% Nsamples=100000;
% AugCounts=floor((Nsamples-length(x_lin_org))/2);
% delta_lin=abs(x_lin_org(1)-x_lin_org(2));
% x_lin_exd_left= linspace(-lim, min(x_lin_org)-delta_lin, AugCounts);
% x_lin_exd_right= linspace(max(x_lin_org)+delta_lin,lim, AugCounts);
% x_lin_exd=[x_lin_exd_left x_lin_org x_lin_exd_right];
% % obtain the excact value of pgo on the extended definition domain
% [pdf_pgo_exd,~,~]=YanFuncLib_Overbound_tmp.two_piece_pdf(x_lin_exd,params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); 
% 
% % set transformation matrix: ecef to enu
% % the tool: https://www.lddgo.net/convert/coordinate-transform
% p_ecef=[-2418235.676841056 , 5386096.899553243 , 2404950.408609563]; % receiver location (ECEF) solved by RTKlib 
% p_lbh=[114.1790017,22.29773881,3];
% p.L=p_lbh(1);p.B=p_lbh(2);p.H=p_lbh(3);
% p.Xp=p_ecef(1);p.Yp=p_ecef(2);p.Zp=p_ecef(3);
% M=YanFuncLib_Unity_tmp.matrix_ecef2enu(p);
% 
% % PL 
% min_s=10000;
% min_s_file='';
% file_error='Data/Least_square_dd_urbandata/error_LSDD.csv';
% error_data = readmatrix(file_error, 'NumHeaderLines', 1);
% PL_pgo_list=zeros(length(error_data),1);
% PL_gaussian_list=zeros(length(error_data),1);
% cal_time_list=zeros(length(error_data),1);
% num_sat_list=zeros(length(error_data),1);
% xerr_list=zeros(length(error_data),1);
% zerr_list=zeros(length(error_data),1);
% gps_week=2238;
% for i=1:length(error_data)
%     % read file
% %     % SPP file
% %     gps_sec=error_data(i,1);
% %     unix_sec= gps_week * 604800.0 + gps_sec + 315964800.0 + 19.0;
% %     xy_error=error_data(i,2);
% %     xyz_error=error_data(i,3);
%     % DGNSS file
%     unix_sec = error_data(i,1);
%     xy_error=error_data(i,4);
%     xyz_error=error_data(i,5);
%     
%     z_error=sqrt(xyz_error^2-xy_error^2);
%     xerr_list(i)=xy_error;
%     zerr_list(i)=z_error;
%     
%     % Open file by filename wildcard 
%     folder = 'Data/Least_square_dd_urbandata/DD_S_matrix';
%     field=num2str(unix_sec);
%     wildcard = fullfile(folder, ['*' field '*']);
%     fileList = dir(wildcard);
%     if isempty(fileList)
%         continue
%     end
%     filename = fullfile(folder, fileList(1).name);
%     S_mat = load(filename);
%     
%     % skip invalid file
% %     % for SPP
% %     if size(S_mat,1)~=6 
% %         continue
% %     end
%     % for DGNSS
%     if size(S_mat,2)<3 
%         continue
%     end
%     
%     % use the positioning part of S matrix
%     S_matp=S_mat(1:3,:);
%     % agumentation
%     S_matpa=zeros(size(S_matp,1)+1,size(S_matp,2)+1);
%     S_matpa(1:size(S_matp,1),1:size(S_matp,2))=S_matp;
%     S_matpa(end,end)=1;
%     % transform
%     S_matTrans=M*S_matpa;
%     % use the core part
%     S_matTransCore=S_matTrans(1:end-1,1:end-1);
%   
% %     scale_list=S_matTransCore(1,:); % related to x error (enu) min_s=0.001410985784173
% %     scale_list=S_mat(2,:); % related to y error (ecef)  min_s=4.974492589715496e-05
%     scale_list=S_mat(3,:); % related to z error (ecef)  min_s=2.313112008120455e-04
%     num_sat_list(i)=length(scale_list);
%     
% %     % for debug
% %     if min_s>min(abs(scale_list))
% %         min_s=min(abs(scale_list));
% %         min_s_file=field;
% %     end
%    
%     % set definition domain of the position domain
%     x_scale=-30:0.01:30;
%     try
%         PHMI=1e-9;
%         [PL_pgo,PL_gaussian,fft_time_all]=YanFuncLib_Overbound_tmp.cal_PL(scale_list,x_scale,std_tsgo,params_pgo,PHMI);
%         PL_pgo_list(i)=PL_pgo;
%         PL_gaussian_list(i)=PL_gaussian;
%         cal_time_list(i)=fft_time_all;
%     catch exception
%         disp('Error!');
%     end
% end
% 
% err_list=zerr_list;
% % save PE and PL for Stanford chart plot
% % save("Urban_PL2.mat","err_list","PL_pgo","PL_gaussian")
% 
% % plot times series of PL and PE
% figure
% yyaxis left
% h1=plot(1:length(error_data),abs(err_list),'k-','linewidth',1.5);
% hold on
% h2=plot(1:length(error_data),abs(PL_gaussian_list),'g-','linewidth',1);
% h3=plot(1:length(error_data),abs(PL_pgo_list),'b-','linewidth',1);
% ylim([0 200])
% ylabel('HPL (m)','FontSize',12);
% yyaxis right
% h4=plot(1:length(error_data),num_sat_list,'c:','linewidth',1);
% ylim([0 55])
% ylabel('Measurement counts');
% xlabel('Time (s)');
% xlim([0 1100])
% ax = gca;
% ax.YAxis(1).Color = 'black';
% ax.YAxis(2).Color = 'black';
% A = legend([h1,h2,h3,h4],'Error','Gaussian','Principal Gaussian','Counts');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% set(A,'FontSize',13.5)
% 
% % summarize computation time
% format long
% A = cal_time_list; % array to be cluster
% B = num_sat_list;  % index of clusers
% uniqueValues = unique(B);
% meanValue_list=groupsummary(A,B,'mean');
% % plot computation time of PL
% figure
% scatter(num_sat_list,cal_time_list,'ko')
% hold on
% plot(uniqueValues,meanValue_list,'r-*');
% ylabel('Computation time (s)');
% xlabel('Number of measurements');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');

%% VPL and VPE series, PL compuation time: Fig. 10a,b (20240406 TAES first revision)
% seed=1234;
% % load GMM
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
%                         30,5,inf,40,'substract median'); % data has human error
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% % counts=length(x_lin_org);
% 
% % gmm fit as emperical
% % [sol,gmm_dist] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
% [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% rng(seed);
% Xgmm_data = random(gmm_dist, 10000);
% % Principal Gaussian overbound (zero-mean)
% alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% alpha_adjust = min(0.5,alpha_adjust);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin_org,gmm_dist,alpha_adjust);
% % check and inflation
% gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin_org,gmm_inflate_pgo,alpha_adjust);
% % gmm emp
% cdf_gmm_inflate_pgo=cdf(gmm_inflate_pgo,x_lin_org')';
% 
% % Two step Gaussian
% % fix bug: 20240319 - use symmetric twp-step bound with defaut param
% [params_tsgo,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin_org);
% 
% % expand the definition domian of the range domain error
% lim=50;
% Nsamples=100000;
% AugCounts=floor((Nsamples-length(x_lin_org))/2);
% delta_lin=abs(x_lin_org(1)-x_lin_org(2));
% x_lin_exd_left= linspace(-lim, min(x_lin_org)-delta_lin, AugCounts);
% x_lin_exd_right= linspace(max(x_lin_org)+delta_lin,lim, AugCounts);
% x_lin_exd=[x_lin_exd_left x_lin_org x_lin_exd_right];
% % obtain the excact value of pgo on the extended definition domain
% [pdf_pgo_exd,~,~]=YanFuncLib_Overbound_tmp.two_piece_pdf(x_lin_exd,params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); 
% 
% % set transformation matrix: ecef to enu
% % the tool: https://www.lddgo.net/convert/coordinate-transform
% p_ecef=[-2418235.676841056 , 5386096.899553243 , 2404950.408609563]; % receiver location (ECEF) solved by RTKlib 
% p_lbh=[114.1790017,22.29773881,3];
% p.L=p_lbh(1);p.B=p_lbh(2);p.H=p_lbh(3);
% p.Xp=p_ecef(1);p.Yp=p_ecef(2);p.Zp=p_ecef(3);
% M=YanFuncLib_Unity_tmp.matrix_ecef2enu(p);
% 
% % PL 
% min_s=10000;
% min_s_file='';
% file_error='Data/Least_square_dd_urbandata/error_LSDD.csv';
% error_data = readmatrix(file_error, 'NumHeaderLines', 1);
% PL_pgo_list=zeros(length(error_data),1);
% PL_gaussian_list=zeros(length(error_data),1);
% cal_time_list=zeros(length(error_data),1);
% num_sat_list=zeros(length(error_data),1);
% xerr_list=zeros(length(error_data),1);
% zerr_list=zeros(length(error_data),1);
% gps_week=2238;
% for i=1:length(error_data)
%     % read file
% %     % SPP file
% %     gps_sec=error_data(i,1);
% %     unix_sec= gps_week * 604800.0 + gps_sec + 315964800.0 + 19.0;
% %     xy_error=error_data(i,2);
% %     xyz_error=error_data(i,3);
%     % DGNSS file
%     unix_sec = error_data(i,1);
%     xy_error=error_data(i,4);
%     xyz_error=error_data(i,5);
%     
%     z_error=sqrt(xyz_error^2-xy_error^2);
%     xerr_list(i)=xy_error;
%     zerr_list(i)=z_error;
%     
%     % Open file by filename wildcard 
%     folder = 'Data/Least_square_dd_urbandata/DD_S_matrix';
%     field=num2str(unix_sec);
%     wildcard = fullfile(folder, ['*' field '*']);
%     fileList = dir(wildcard);
%     if isempty(fileList)
%         continue
%     end
%     filename = fullfile(folder, fileList(1).name);
%     S_mat = load(filename);
%     
%     % skip invalid file
% %     % for SPP
% %     if size(S_mat,1)~=6 
% %         continue
% %     end
%     % for DGNSS
%     if size(S_mat,2)<3 
%         continue
%     end
%     
%     % use the positioning part of S matrix
%     S_matp=S_mat(1:3,:);
%     % agumentation
%     S_matpa=zeros(size(S_matp,1)+1,size(S_matp,2)+1);
%     S_matpa(1:size(S_matp,1),1:size(S_matp,2))=S_matp;
%     S_matpa(end,end)=1;
%     % transform
%     S_matTrans=M*S_matpa;
%     % use the core part
%     S_matTransCore=S_matTrans(1:end-1,1:end-1);
%   
% %     scale_list=S_matTransCore(1,:); % related to x error (enu) min_s=0.001410985784173
% %     scale_list=S_mat(2,:); % related to y error (ecef)  min_s=4.974492589715496e-05
%     scale_list=S_mat(3,:); % related to z error (ecef)  min_s=2.313112008120455e-04
%     num_sat_list(i)=length(scale_list);
%     
% %     % for debug
% %     if min_s>min(abs(scale_list))
% %         min_s=min(abs(scale_list));
% %         min_s_file=field;
% %     end
%    
%     % set definition domain of the position domain
%     x_scale=-30:0.01:30;
% %     try
%         PHMI=1e-9;
%         [PL_pgo,PL_gaussian,fft_time_all]=YanFuncLib_Overbound_tmp.cal_PL(scale_list,x_scale,params_tsgo,params_pgo,PHMI,'ob_discrete');
%         PL_pgo_list(i)=PL_pgo;
%         PL_gaussian_list(i)=PL_gaussian;
%         cal_time_list(i)=fft_time_all;
% %     catch exception
% %         disp('Error!');
% %     end
% end
% 
% err_list=zerr_list;
% % save PE and PL for Stanford chart plot
% % save("Urban_PL2.mat","err_list","PL_pgo","PL_gaussian")
% 
% % plot times series of PL and PE
% figure
% % yyaxis left
% h1=plot(1:length(error_data),abs(err_list),'k-','linewidth',1.5);
% hold on
% h2=plot(1:length(error_data),abs(PL_gaussian_list),'g-','linewidth',1);
% h3=plot(1:length(error_data),abs(PL_pgo_list),'b-','linewidth',1);
% ylim([0 200])
% ylabel('Vertical protection level (m)','FontSize',12);
% % yyaxis right
% % h4=plot(1:length(error_data),num_sat_list,'c:','linewidth',1);
% % ylim([0 55])
% % ylabel('Measurement counts');
% xlabel('Time (s)');
% xlim([0 1100])
% % ax = gca;
% % ax.YAxis(1).Color = 'black';
% % ax.YAxis(2).Color = 'black';
% % A = legend([h1,h2,h3,h4],'Error','Gaussian','Principal Gaussian','Counts');
% A = legend([h1,h2,h3],'Error','Gaussian','Principal Gaussian');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% set(A,'FontSize',13.5)
% % 
% % summarize computation time
% format long
% A = cal_time_list; % array to be cluster
% B = num_sat_list;  % index of clusers
% uniqueValues = unique(B);
% meanValue_list=groupsummary(A,B,'mean');
% % plot computation time of PL
% figure
% scatter(num_sat_list,cal_time_list,'ko')
% hold on
% plot(uniqueValues,meanValue_list,'r-*');
% ylabel('Computation time (s)');
% xlabel('Number of measurements');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');

% % show statistics
% PL_dff = abs((abs(PL_pgo_list)-abs(PL_gaussian_list))./abs(PL_gaussian_list));
% min(PL_dff)
% max(PL_dff)
% mean(PL_dff)
% median(PL_dff)

%% Paper-Stanford chart: Fig. 10c,d
% % load('Urban_PL_revision_20240409.mat')
% VPE = error_list;VPL_Gaussian=PL_gaussian_list;VPL_pgo=PL_pgo_list;
% 
% E=abs(VPE');
% PL=abs(VPL_Gaussian');
% % PL=abs(VPL_pgo');
% % Figure position
% L = 800;
% H = 700;
% Position = get(0,'screensize');
% Position = [(Position(3)-L)/2 (Position(4)-H)/2 L H];
% % Figure
% Figure = ...
%     figure('Color',      'w',...
%            'NumberTitle','Off',...
%            'Name',       '');
% % Axes       
% Axes = ...
%     axes('Parent',   Figure,...
%          'DataAspect',[1 1 1]);
% AlertLimit = []; CategoryLimit = [];
% Strings = {'NO #1','NO #2'};
% 
% StanfordDiagram('Axes',            Axes,...
%                 'Step',            2,...
%                 'Maximum',         150,...
%                 'AlertLimit',      AlertLimit,...
%                 'CategoryLimit',   CategoryLimit,...
%                 'Scale',           'linear',...
%                 'Strings',         Strings,...
%                 'StringFcns',      repmat({@Display},1,7),...
%                 'FontSize',        9,...
%                 'PositionError',   E,...
%                 'ProtectionLevel', PL); 
% 
% drawnow();
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');

%% Demonstration of discretization and PL calculation: Fig. 3 (20240507)
% x_left = -3:0.3:-2.4;
% stem(x_left,normpdf(x_left),'r');
% hold on
% stem(-2.1,normpdf(-2.1),'k');
% x_right = 2.1:0.3:3;
% stem(x_right,normpdf(x_right),'k');
% x_center = -0.6:0.3:0.6;
% stem(x_center,normpdf(x_center,0,3.5),'k');
% 
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% set(A,'FontSize',13.5)
% ylabel('PMF')
% xlabel('Error (m)')
% %  ylim([-0.05,0.15])
% xlim([-3.2,3.2])
% grid on

%% Paper-alpha effects: Fig. 8 (TAES draft)
% seed=1234;
% % load Data
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD({'Data/mnav_zmp1_jan/mergedRefJan.mat'},...
%     30,5,'2020-01-01','2020-01-31 23:59:59',10,'substract median'); % data has human error
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% counts=length(x_lin);
% 
% % Two step Gaussian
% [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound(Xdata,x_lin);
% % Gaussian Pareto
% [params_gpo,pdf_gpo,cdf_gpo]=YanFuncLib_Overbound_tmp.Gaussian_Pareto_bound(Xdata,x_lin);
% 
% % Principal Gaussian overbound (zero-mean)
% % Type | Ele. | Inflate  | alpha |
% % Ref  | 30-35| 1,   1.15|  0.7  |
% gmm_dist_raw=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_raw,1,1.15) 
% 
% % set coloar map and length of data array
% % log scale cdf (left side)
% figure;
% semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4,'DisplayName','Sample dist.');
% hold on
% semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'g','LineWidth',2,'DisplayName','Gaussian');
% semilogy(x_lin,cdf_gpo,'r','LineWidth',2,'DisplayName','Gaussian-Pareto');
% mapName = 'jet';
% counts = 2;
% % generate color map
% cmap = colormap(mapName);
% step=floor(length(cmap)/counts)+1;
% % Intercept a color map of specified length
% colorArray  = cmap(1:step:end, :);
% for i=1:counts
%     alpha=i*0.05+0.5;
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha); 
%     semilogy(x_lin,cdf_pgo,'LineWidth',1,'DisplayName',num2str(alpha),'color',colorArray(i,:));
% end
% xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend();
% set(A,'FontSize',13.5)
% grid on

%% Paper-alpha effects: Fig. 11 (20240408 TAES first revision)
% seed=1234;
% ele=30;
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% % load Data
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_RefDD(...
%     {'Data/mnav_zmp1_halfyear_20240410/mergedRefhalfyear.mat','Data/mnav_zmp1_halfyear_2nd_20240410/mergedRefhalfyear2nd.mat'},...
%     ele,5,'2020-01-01 00:00:00','2020-01-31 23:59:59',10,'mirror data','substract median');
% % [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
% %                         30,5,inf,40,'substract median'); % data has human error
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% counts=length(x_lin);
% 
% % gmm fit as emperical
% % [sol,gmm_dist] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
% [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% rng(seed);
% Xgmm_data = random(gmm_dist, 10000);
% kurtosis(Xgmm_data);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';
% 
% % set coloar map and length of data array
% % log scale cdf (left side)
% figure;
% % semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4,'DisplayName','Sample dist.');
% semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 4,'DisplayName','Sample dist.');
% hold on
% % % GMM plot
% % semilogy(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % % PGO plot
% % semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% mapName = 'jet';
% counts = 22;
% % generate color map
% cmap = colormap(mapName);
% step=floor(length(cmap)/counts)+1;
% % Intercept a color map of specified length
% colorArray  = cmap(1:step:end, :);
% for i=1:counts
%     if i==1
%         thr=0.01;
%     else
%         thr=(i-1)*0.05;
%     end
% %     thr = 0.01*i;
%     % Principal Gaussian overbound (zero-mean)
%     alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist,thr);
%     alpha_adjust = min(0.5,alpha_adjust);
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
%     % check and inflation
% %     gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
% %     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
% %     semilogy(x_lin,cdf_pgo,'LineWidth',1,'DisplayName',num2str(thr),'color',colorArray(i,:));
%     semilogy(x_lin,1-cdf_pgo,'LineWidth',1,'DisplayName',num2str(thr),'color',colorArray(i,:));
% end
% % xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend();
% set(A,'FontSize',13.5)
% grid on

%% Bias effects and paired Principal Gaussian overbound: future work
% seed=1234;
% % load GMM
% [Xdata,x_lin,pdf_data,cdf_data,~]=YanFuncLib_Overbound_tmp.load_GMM_bias(seed);
% gmm_dist=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';
% 
% % Principal Gaussian overbound (zero-mean)
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,0.7);
% 
% figure
% subplot(1,2,1)
% histogram(Xdata,'normalization','pdf');
% hold on
% plot(x_lin,pdf_emp,'k--','LineWidth',2);
% 
% subplot(1,2,2)
% [ecdf_data,x_ecdf]=ecdf(Xdata);
% plot(x_ecdf,ecdf_data,'k','LineWidth',1);
% hold on
% plot(x_lin,cdf_emp,'k--','LineWidth',2);
% plot(x_lin,cdf_pgo,'r','LineWidth',2);
% 
% figure;
% histogram(Xdata,'BinWidth',0.2,'FaceAlpha',1,'FaceColor','b');
% hold on
% histogram(Xdata(Xdata>=median(Xdata)),'BinWidth',0.2,'FaceColor','y');
% histogram(Xdata(Xdata<=median(Xdata)),'BinWidth',0.2,'FaceAlpha',1,'FaceColor','b');
% xline(median(Xdata),'k--','LineWidth',2);
% % Principal Gaussian overbound (left)
% Xmedian=median(Xdata);
% Xleft=Xdata(Xdata<Xmedian);
% Xleft_recon=[Xleft;2*Xmedian-Xleft;Xmedian];
% gmm_dist_left=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xleft_recon-mean(Xleft_recon));
% gmm_dist_left=YanFuncLib_Overbound_tmp.inflate_GMM(gmm_dist_left,1,1.1)
% [params_pgo_left, pdf_pgo_left, cdf_pgo_left]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xleft_recon-mean(Xleft_recon),x_lin,gmm_dist_left,0.6);
% counts=length(x_lin);
% plot(x_lin(1:floor(counts/2))+mean(Xleft_recon),pdf_pgo_left(1:floor(counts/2))*length(Xdata)*0.2,'r','LineWidth',2);
% 
% figure;
% histogram(Xleft_recon,'BinWidth',0.2,'FaceColor','b');
% hold on
% Xleft_recon_center=Xleft_recon-Xmedian;
% histogram(Xleft_recon_center,'BinWidth',0.2,'FaceColor','g');
% plot(x_lin,pdf_pgo_left*length(Xdata)*0.2,'r','LineWidth',2);


%% Pape-Overbounding compare all angle: Fig. 13
% seed=1234;
% use_subplots=true;
% load('Data/mnav_zmp1_halfyear_20240410/mergedRefhalfyear.mat');
% load('Data/mnav_zmp1_halfyear_2nd_20240410/mergedRefhalfyear2nd.mat');
% tmp = vertcat(mergedRefhalfyear, mergedRefhalfyear2nd);
% % mergedRefAll = tmp;
% filter_date = tmp.datetime>'2020-01-01' & tmp.datetime<'2021-02-01';
% mergedRefAll = tmp(filter_date,:);
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% for fig_cnt=1:2
%     figure;
%     item=1;
%     for ele = 15+(fig_cnt-1)*6*5:5:15+fig_cnt*6*5-5
%         filter_ele=(mergedRefAll.U2I_Elevation>=ele & mergedRefAll.U2I_Elevation<=ele+5); 
%         filter_err=(mergedRefAll.doubledifferenced_pseudorange_error>=-10 & mergedRefAll.doubledifferenced_pseudorange_error<=10); 
%         Xdata=mergedRefAll.doubledifferenced_pseudorange_error(filter_ele & filter_err);
%         Xdata = -Xdata;
%         % shift the distribution to obtian zero-median
%         Xdata = Xdata - median(Xdata);
% %         Xdata = unique(Xdata);
%         if length(Xdata)<2500*2
%             continue
%         end
%         
%         % prevent too many data points
%         Nsamples=length(Xdata);
%         if Nsamples<15000
%             Nsamples=15000; 
%         end
%         if Nsamples>50000
%             Nsamples=50000; 
%         end
%         
%         lim=max(-min(Xdata),max(Xdata));
%         x_lin = linspace(-lim, lim, Nsamples);
%         str_title =['Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ'];
%         
%         % ecdf
%         [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
%         counts=length(x_lin);
%         
%         % gmm fit as emperical
%         % [sol,gmm_dist] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
%         [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
%         rng(seed);
%         Xgmm_data = random(gmm_dist, 10000);
%         kurtosis(Xgmm_data);
%         pdf_emp=pdf(gmm_dist,x_lin')';
%         cdf_emp=cdf(gmm_dist,x_lin')';
%         
%         % PGO
%         alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xdata,gmm_dist);
%         alpha_adjust = min(0.5,alpha_adjust);
%         [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
%         % check and inflation
%         gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
%         [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
%         % gmm emp
%         cdf_gmm_inflate_pgo=cdf(gmm_inflate_pgo,x_lin')';
%     
%         % Two step Gaussian
%         % fix bug: 20240319 - use symmetric twp-step bound with defaut param
%         [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin);
%         % Gaussian Pareto
%         [params_gpo,pdf_gpo,cdf_gpo]=YanFuncLib_Overbound_tmp.Gaussian_Pareto_bound(Xdata,x_lin);
%         
%         % save data file
%         save(['C:\Users\Administrator\Desktop\Test_data\1year_',num2str(ele),'deg_20240410'],...
%             'x_lin_ecdf','ecdf_data',...
%             'x_lin','params','cdf_left_tsgo','cdf_right_tsgo',...
%             'cdf_gpo',...
%             'cdf_emp',...
%             'cdf_pgo',...
%             'Xdata','str_title')
%     end
% end
% 
% % % manually draw
% % ele=15
% % file = ['C:\Users\Administrator\Desktop\Test_data\1year_',num2str(ele),'deg_20240410'];
% % str_title =['Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ'];
% % load(file)
% % % ecdf plot
% % h1=semilogy(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 3);
% % hold on
% % % Two step Gaussian plot
% % h21=semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'go-','LineWidth',1.5,'MarkerSize', 3,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% % % Gaussian Pareto plot
% % h3=semilogy(x_lin,cdf_gpo,'r','LineWidth',1);
% % % GMM plot
% % h4=semilogy(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % % PGO plot
% % h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',0.5,'MarkerSize', 3,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % xlim([min(Xdata)*1.2,max(Xdata)*0.5]);
% % ylim([1e-7,1]);
% % title(str_title);
% % set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% % % A = legend([h1,h21,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SE');
% % % set(A,'FontSize',13.5)
% % grid on
% % 
% % % ecdf plot
% % h1=semilogy(x_lin_ecdf,1-ecdf_data,'kx-','LineWidth',1,'MarkerSize', 3);
% % hold on
% % % Two step Gaussian plot
% % h24=semilogy(x_lin(params.idx+1:end),1-cdf_right_tsgo(params.idx+1:end),'go-','LineWidth',1.5,'MarkerSize', 3,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% % % Gaussian Pareto plot
% % h3=semilogy(x_lin,1-cdf_gpo,'r','LineWidth',1);
% % % GMM plot
% % h4=semilogy(x_lin,1-cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % % PGO plot
% % h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',0.5,'MarkerSize', 3,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % xlim([min(Xdata)*0.5,max(Xdata)*1.2]);
% % ylim([1e-7,1]);
% % xlabel('Error (m)');
% % title(str_title);
% % set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% % grid on

%% Conf. paper-PNT 2024: revision (20240413)
% figure
% seed=1234;
% use_subplots=false;
% ele = 60;
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% % load Data
% [Xdata,x_lin,pdf_data,cdf_data]=YanFuncLib_Overbound_tmp.load_NIG();
% % [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/urban_dd_0816/mergeurbandd.mat'},...
% %                         30,5,inf,40,'substract median'); % data has human error
% 
% % ecdf
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% 
% % gmm fit as emperical
% [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% rng(seed);
% Xgmm_data = random(gmm_dist, 10000);
% kurtosis(Xgmm_data);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';
% 
% % Principal Gaussian overbound (zero-mean)
% alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% alpha_adjust = min(0.5,alpha_adjust);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
% % check and inflation
% gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
% % gmm emp
% cdf_gmm_inflate_pgo=cdf(gmm_inflate_pgo,x_lin')';
% 
% % Two step Gaussian
% % fix bug: 20240319 - use symmetric twp-step bound with defaut param
% [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin);
% 
% % Gaussian Pareto
% [params_gpo,pdf_gpo,cdf_gpo]=YanFuncLib_Overbound_tmp.Gaussian_Pareto_bound(Xdata,x_lin);
% 
% % qq plot
% if ~use_subplots
%     figure
% else
%     subplot(2,2,1)
% end
% Xnorm=randn(1,length(Xdata));
% h=qqplot(Xdata,Xnorm);
% xlabel('Quantiles of error distribution (m)');
% ylabel('Standard normal quantile (m)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% grid on
% 
% % cdf plot
% if ~use_subplots
%     figure
% else
%     subplot(2,2,2)
% end
% % ecdf plot
% h1=plot(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6,'MarkerIndices',1:floor(length(x_lin_ecdf)/30):length(x_lin_ecdf));
% hold on
% % Two step Gaussian plot
% h21=plot(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'go-','LineWidth',1.5,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% h24=plot(x_lin(params.idx+1:end),cdf_right_tsgo(params.idx+1:end),'g--','LineWidth',1.5);
% % Gaussian Pareto plot
% h3=plot(x_lin,cdf_gpo,'r','LineWidth',2);
% % GMM plot
% h4=plot(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % PGO plot
% h5=plot(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/24):length(x_lin));
% xlabel('Error (m)');
% ylabel('CDF');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h21,h24,h3,h4,h5],'Sample dist.','Two-step Gaussian (L)','Two-step Gaussian (R)','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SE');
% set(A,'FontSize',13.5)
% grid on
% 
% % log scale cdf plot (left side) 
% if ~use_subplots
%     figure
% else
%     subplot(2,2,3)
% end
% % ecdf plot
% h1=semilogy(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6);
% hold on
% % Two step Gaussian plot
% h21=semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'go-','LineWidth',1.5,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% % Gaussian Pareto plot
% h3=semilogy(x_lin,cdf_gpo,'r','LineWidth',2);
% % GMM plot
% h4=semilogy(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % PGO plot
% h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % yline(0.5);
% xlim([min(x_lin)*1.2,max(x_lin)*0.5]);
% ylim([1e-7,1]);
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h21,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SE');
% set(A,'FontSize',13.5)
% grid on
% 
% % log scale cdf plot (right side)
% if ~use_subplots
%     figure
% else
%     subplot(2,2,4)
% end
% % ecdf plot
% h1=semilogy(x_lin_ecdf,1-ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6);
% hold on
% % Two step Gaussian plot
% h24=semilogy(x_lin(params.idx+1:end),1-cdf_right_tsgo(params.idx+1:end),'go-','LineWidth',1.5,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/28):length(x_lin));
% % Gaussian Pareto plot
% h3=semilogy(x_lin,1-cdf_gpo,'r','LineWidth',2);
% % GMM plot
% h4=semilogy(x_lin,1-cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % PGO plot
% h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% % yline(0.5);
% xlim([min(x_lin)*0.5,max(x_lin)*1.2]);
% ylim([1e-7,1]);
% xlabel('Error (m)');
% ylabel('CCDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h24,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SW');
% set(A,'FontSize',13.5)
% grid on


function d = Display(e,s)
    if s > 0.01
        d = sprintf('%s(%.3f%%)',char(10),100*s);
    else
        d = sprintf('%s(%u epochs)',char(10),e);
    end
end
