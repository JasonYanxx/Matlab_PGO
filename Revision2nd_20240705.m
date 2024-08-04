addLibPathInit();
YanFuncLib_Overbound_tmp=YanFuncLib_Overbound;
YanFuncLib_Unity_tmp=YanFuncLib_Unity;
%% Urban overbounding compare
% figure
% seed=1234;
% use_subplots=false;
% ele = 30;
% void_ratio = 0.5/100;% 0.1% for urban; 1% for ref
% % load Data
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat'},...
%                         ele,5,inf,35,'substract median','TAES_2nd_complementary');
% str_title =['Urban DGNSS Errors (Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ)'];
% 
% % ecdf
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% % x_lin_ecdf=sort(Xdata);
% % ecdf_data = linspace(1/length(x_lin_ecdf), 1-1/length(x_lin_ecdf), length(x_lin_ecdf))'; % a better way
% 
% % gmm fit as emperical
% %[sol,gmm_dist] = YanFuncLib_Overbound_tmp.opfit_GMM_zeroMean(Xdata);
% [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% rng(seed);
% Xgmm_data = random(gmm_dist, 10000);
% kurtosis(Xgmm_data);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';
% 
% % Principal Gaussian overbound (zero-mean)
% % alpha_adjust =  0;
% % lpha_adjust =  exp(3-kurtosis(Xdata));
% % alpha_adjust =  (3/kurtosis(Xdata));
% % alpha_adjust = 3/kurtosis(Xgmm_data)*0.5;
% % alpha_adjust=0.15;
% % alpha_adjust1=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% % alpha_adjust2=YanFuncLib_Overbound_tmp.find_alpha(-Xgmm_data,gmm_dist);
% % alpha_adjust = min([0.5,alpha_adjust1,alpha_adjust2]);
% alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% alpha_adjust = min(0.5,alpha_adjust);
% [params_pgo, pdf_pgo, cdf_pgo_pre]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
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
% ylim([-5,5])
% xlabel('Quantiles of error distribution (m)');
% ylabel('Standard normal quantile (m)');
% title(str_title);
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
% title(str_title);
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
% ylim([1e-5,1])
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% title(str_title);
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
% ylim([1e-5,1])
% xlabel('Error (m)');
% ylabel('CCDF (log scale)');
% title(str_title);
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h24,h3,h4,h5],'Sample dist.','Two-step Gaussian','Gaussian-Pareto','BGMM fitting','Principal Gaussian','Location','SW');
% set(A,'FontSize',13.5)
% grid on

%% Urban DGNSS error against SNR and Ele: Fig.8
% load('Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat');
% mergedurbandd = mergedTSTJun28onehour; 
% filter_datetime=(mergedurbandd.datetime>"2024-06-28 06:01:59");
% ufilter_datetime=(mergedurbandd.datetime>"2024-06-28 06:55:51" & mergedurbandd.datetime<"2024-06-28 06:56:47" );
% mergedurbandd = mergedurbandd(filter_datetime & ~ufilter_datetime,:);
% 
% figure
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

%% Urban-bias effects
% load('Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat');
% mergedurbandd = mergedTSTJun28onehour;
% seed=1234;
% filter_SNR=(mergedurbandd.U2I_SNR>=35); 
% filter_datetime=(mergedurbandd.datetime>"2024-06-28 06:01:59");
% ufilter_datetime=(mergedurbandd.datetime>"2024-06-28 06:55:51" & mergedurbandd.datetime<"2024-06-28 06:56:47" );
% mergedRefAll  = mergedurbandd(filter_datetime & ~ufilter_datetime & filter_SNR,:);
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% 
% ele = 55;
% filter_ele=(mergedRefAll.U2I_Elevation>=ele & mergedRefAll.U2I_Elevation<=ele+5); 
% Xdata=mergedRefAll.doubledifferenced_pseudorange_error(filter_ele);
% 
% % prevent too many data points
% Nsamples=length(Xdata);
% if Nsamples<15000
%     Nsamples=15000; 
% end
% if Nsamples>50000
%     Nsamples=50000; 
% end
% 
% lim=max(-min(Xdata),max(Xdata));
% x_lin = linspace(-lim, lim, Nsamples);
% str_title =['Urban DGNSS Errors (Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ)'];
% 
% % ecdf
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
% % PGO
% alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xdata,gmm_dist);
% alpha_adjust = min(0.5,alpha_adjust);
% [params_pgo, pdf_pgo, cdf_pgo_pre]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
% % check and inflation
% gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
% % gmm emp
% cdf_gmm_inflate_pgo=cdf(gmm_inflate_pgo,x_lin')';
% 
% figure;
% h1=plot(x_lin_ecdf,ecdf_data,'kx-','LineWidth',1,'MarkerSize', 6,'MarkerIndices',1:floor(length(x_lin_ecdf)/30):length(x_lin_ecdf));
% hold on
% % PGO plot
% h51=plot(x_lin,cdf_pgo_pre,'rd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/24):length(x_lin));
% % PGO plot
% h52=plot(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/24):length(x_lin));
% xlabel('Error (m)');
% ylabel('CDF');
% title(str_title);
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h51,h52],'Sample dist.','PGO (without inflation)','PGO (sigma inflation)');
% set(A,'FontSize',13.5)
% yline(0.5,'HandleVisibility','off')
% grid on

%% protection level
% seed = 1234;
% %---------------generate overbound---------------%
% ele = 30;
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat'},...
%                         ele,5,inf,35,'substract median','TAES_2nd_complementary');
% % gmm fit as emperical
% [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% rng(seed);
% Xgmm_data = random(gmm_dist, 10000);
% % Principal Gaussian overbound (zero-mean)
% alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% alpha_adjust = min(0.5,alpha_adjust);
% [params_pgo, pdf_pgo, cdf_pgo_pre]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
% % check and inflation
% gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
% % Two step Gaussian
% [params_tsgo,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin);
% 
% %---------------define location and projection---------------%
% ref_xyz = [-2414266.9197, 5386768.9868, 2407460.0314]; % HKSC from Hong Kong Geodetic Survey Services
% % GTusr_xyz = [-2418235.676841056, 5386096.899553243, 2404950.408609563]; % receiver location (ECEF) solved by RTKlib
% GTusr_xyz = [-2418225.8846   5386100.2924   2404950.2195];
% M_ecef2enu = YanFuncLib_Unity_tmp.get_trans_mat_ecef2enu(GTusr_xyz); % prepare for ENU results
% 
% %---------------load observations---------------%
% load('Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat');
% mergedurbandd = mergedTSTJun28onehour;
% filter_datetime=(mergedurbandd.datetime>"2024-06-28 06:01:59");
% ufilter_datetime=(mergedurbandd.datetime>"2024-06-28 06:55:51" & mergedurbandd.datetime<"2024-06-28 06:56:47" );
% filter_ele = (mergedurbandd.U2I_Elevation > 30);
% filter_SNR = (mergedurbandd.U2I_SNR > 35);
% filter_GPS = (mergedurbandd.Master_sat_id <=32);
% filter_GLONASS = (mergedurbandd.Master_sat_id >32 & mergedurbandd.Master_sat_id <=82);
% filter_BDS = (mergedurbandd.Master_sat_id >82);
% mergedurbandd_sel = mergedurbandd(filter_ele & filter_SNR & filter_datetime & ~ufilter_datetime,:);
% tData_sorted = sortrows(mergedurbandd_sel,1);
% all_epoches = sort(unique(tData_sorted.datetime));
%     
% %---------------Calculate position and bound---------------%
% T = 0.01; % SAMPLE INTERVAL
% error_list = zeros(length(all_epoches),1);
% PL_pgo_list=zeros(length(all_epoches),1);
% PL_gaussian_list=zeros(length(all_epoches),1);
% cal_time_list=zeros(length(all_epoches),1);
% num_sat_list=zeros(length(all_epoches),1);
% 
% for i=1:length(all_epoches)
%     i
%     epoch = all_epoches(i);
%     % select epoch
%     filter_date= (tData_sorted.datetime==epoch);
%     Xdata=tData_sorted(filter_date,:);
%     % meas
%     meas = (Xdata.user2master_pseudorange - Xdata.ref2master_pseudorange) ...
%           - (Xdata.user2target_pseudorange - Xdata.ref2target_pseudorange); % (rho_s_r1 - rho_s_r2) - (rho_i_r1 - rho_i_r2)
%     meas = -meas;
%     % sat position
%     m_sv_pos = [Xdata.master_x,Xdata.master_y,Xdata.master_z];
%     s_sv_pos = [Xdata.target_x,Xdata.target_y,Xdata.target_z];
%     % other info.
%     init_state=zeros(3,1);
% 
%     if size(s_sv_pos,1)>=3
%         % LS positioning
%         meas_std = ones(size(meas));
%         [eWLSSolution,prn_res,ErrorECEF,G,eDeltaPos,eDeltaPr] = YanFuncLib_Unity_tmp.WeightedLeastSquareDD(GTusr_xyz, init_state, meas, meas_std, m_sv_pos,s_sv_pos,ref_xyz);
%         % position error at ENU
%         PE_enu = YanFuncLib_Unity_tmp.trans_pos_ecef2enu(eWLSSolution(1:3),M_ecef2enu) ...
%                         -YanFuncLib_Unity_tmp.trans_pos_ecef2enu(GTusr_xyz',M_ecef2enu);
%         error_list(i) = PE_enu(3);
%         % get scale list
%         S = inv(G'*G)*G';
%         S_enu = YanFuncLib_Unity_tmp.trans_solmat_ecef2enu(S,M_ecef2enu);
%         scale_list=S_enu(3,:); % related to z error (enu)  
%         num_sat_list(i)=length(scale_list);   
%         % set definition domain of the position domain
%         x_scale=-30:T:30;
% %         try
%             PHMI=1e-9;
%             [PL_pgo,PL_gaussian,fft_time_all]=YanFuncLib_Overbound_tmp.cal_PL(scale_list,x_scale,params_tsgo,params_pgo,PHMI,'ob_discrete');
%             PL_pgo_list(i)=PL_pgo;
%             PL_gaussian_list(i)=PL_gaussian;
%             cal_time_list(i)=fft_time_all;
% %         catch exception
% %             disp('Error!');
% %         end
%     else
%         % invalid positioning
%         a=0;
%     end
%     if i==166
%         a=0;
%     end
% end
% % 
% % %---------------plot times series of PL and PE---------------%
% % save(['Urban_PL_revision_20240706_T_',num2str(T),'.mat'],'error_list','PL_gaussian_list','PL_pgo_list','cal_time_list','num_sat_list')
% % figure
% % h1=plot(1:length(error_list),abs(error_list),'k-o','linewidth',1,'DisplayName','VPE');
% % hold on
% % h2=plot(1:length(error_list),abs(PL_gaussian_list),'g-','linewidth',0.5,'DisplayName','VPL by two-step Gaussian');
% % h3=plot(1:length(error_list),abs(PL_pgo_list),'b:','linewidth',1,'DisplayName','VPL by Principal Gaussian');
% % % ylim([0 200])
% % ylabel('Vertical protection level (m)','FontSize',12);
% % xlabel('Time (s)')
% % set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% % box off
% % %--------------- show statistics---------------%
% % PL_dff = abs((abs(PL_pgo_list)-abs(PL_gaussian_list))./abs(PL_gaussian_list));
% % min(PL_dff)
% % max(PL_dff)
% % mean(PL_dff)
% % median(PL_dff)
% % %---------------summarize and plot computation time---------------%
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
%% show T-effects
% figure
% load('Urban_PL_revision_20240706_T_0.01.mat')
% plot(1:length(error_list),abs(error_list),'k-','linewidth',1.5,'DisplayName','VPE');
% hold on
% for T = [0.01,0.05,2,5,10]
%     load(['Urban_PL_revision_20240706_T_',num2str(T),'.mat'])
%     plot(1:length(error_list),abs(PL_pgo_list),'-','linewidth',1,'DisplayName',['T=',num2str(T),'m']);
% end
% ylabel('Vertical protection level (m)','FontSize',12);
% xlabel('Time (s)')
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
%% show histogram
% load('Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat');
% mergedurbandd = mergedTSTJun28onehour;
% filter_SNR=(mergedurbandd.U2I_SNR>=35); 
% filter_datetime=(mergedurbandd.datetime>"2024-06-28 06:01:59");
% ufilter_datetime=(mergedurbandd.datetime>"2024-06-28 06:55:51" & mergedurbandd.datetime<"2024-06-28 06:56:47" );
% 
% figure
% item=1;
% % for ele = [5,10,15,20,25,30,35,40,45,50,55,60]
% for ele = [30,35,40,45,50,55]
%     filter_ele=(mergedurbandd.U2I_Elevation>=ele & mergedurbandd.U2I_Elevation<=ele+5); 
%     Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_datetime & ~ufilter_datetime & filter_ele & filter_SNR);
%     subplot(2,3,item)
%     histogram(Xdata,'normalization','pdf')
% %     xlim([-8,8])
% %     ylim([-0.01,0.7])
%     str1 = ['Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ'];
%     str2 = ['Min: ',num2str(min(Xdata)),' ,Max:',num2str(max(Xdata))];
%     str3 = ['Mean: ',num2str(mean(Xdata)),' ,Median:',num2str(median(Xdata)),' ,Amount:',num2str(length(Xdata))];
%     str4 = ['Mean: ',num2str(mean(Xdata)),' ,Amount:',num2str(length(Xdata))];
%     title({str1,str4});
%     xlabel(['(',char('a'+item),')'])
%     set(gca, 'FontSize', 13,'FontName', 'Times New Roman');
%     item=item+1;
% end
%% Paper-alpha effects: Fig. 11 
% 
% figure
% seed=1234;
% use_subplots=false;
% ele = 30;
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% % load Data
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat'},...
%                         ele,5,inf,35,'substract median','TAES_2nd_complementary');
% str_title =['Urban DGNSS Errors (Elev.: ',num2str(ele),'\circ \sim ',num2str(ele+5),'\circ)'];
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
% semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4,'DisplayName','Sample dist.');
% % semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 4,'DisplayName','Sample dist.');
% hold on
% % % GMM plot
% % semilogy(x_lin,cdf_emp,'ms--','LineWidth',1,'MarkerFaceColor','m','MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/26):length(x_lin));
% % % PGO plot
% % semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% mapName = 'jet';
% counts = 32;
% % generate color map
% cmap = colormap(mapName);
% step=2;
% % Intercept a color map of specified length
% colorArray  = cmap(1:step:end, :);
% for i=1:counts
%     i
%     thr = 0.01*i;
%     % Principal Gaussian overbound (zero-mean)
%     alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist,thr);
%     alpha_adjust = min(0.5,alpha_adjust);
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
%     % check and inflation
% %     gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
% %     [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
%     semilogy(x_lin,cdf_pgo,'LineWidth',1,'DisplayName',num2str(thr),'color',colorArray(i,:));
% %     semilogy(x_lin,1-cdf_pgo,'LineWidth',1,'DisplayName',num2str(thr),'color',colorArray(i,:));
% end
% % xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend();
% set(A,'FontSize',13.5)
% ylim([1e-4,1])
% grid on

%% Monte Carlo PL caculation time
% seed = 1234;
% %---------------generate overbound---------------%
% ele = 30;
% void_ratio = 1/100;% 0.1% for urban; 1% for ref
% [Xdata,x_lin,pdf_data]=YanFuncLib_Overbound_tmp.load_UrbanDD({'Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat'},...
%                         ele,5,inf,35,'substract median','TAES_2nd_complementary');
% % gmm fit as emperical
% [gmm_dist]=YanFuncLib_Overbound_tmp.gene_GMM_EM_zeroMean(Xdata);
% rng(seed);
% Xgmm_data = random(gmm_dist, 10000);
% % Principal Gaussian overbound (zero-mean)
% alpha_adjust=YanFuncLib_Overbound_tmp.find_alpha(Xgmm_data,gmm_dist);
% alpha_adjust = min(0.5,alpha_adjust);
% [params_pgo, pdf_pgo, cdf_pgo_pre]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha_adjust);
% % check and inflation
% gmm_inflate_pgo=YanFuncLib_Overbound_tmp.inflate_PGO_gmm(params_pgo,alpha_adjust,gmm_dist,Xdata,void_ratio);
% [params_pgo, pdf_pgo, cdf_pgo]=YanFuncLib_Overbound_tmp.Principal_Gaussian_bound(Xdata,x_lin,gmm_inflate_pgo,alpha_adjust);
% % Two step Gaussian
% [params_tsgo,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFuncLib_Overbound_tmp.two_step_bound_practical(Xdata,x_lin);
% 
% %---------------define location and projection---------------%
% ref_xyz = [-2414266.9197, 5386768.9868, 2407460.0314]; % HKSC from Hong Kong Geodetic Survey Services
% % GTusr_xyz = [-2418235.676841056, 5386096.899553243, 2404950.408609563]; % receiver location (ECEF) solved by RTKlib
% GTusr_xyz = [-2418225.8846   5386100.2924   2404950.2195];
% M_ecef2enu = YanFuncLib_Unity_tmp.get_trans_mat_ecef2enu(GTusr_xyz); % prepare for ENU results
% 
% %---------------load observations---------------%
% load('Data/TST_Jun28_onehour/mergedTSTJun28onehour.mat');
% mergedurbandd = mergedTSTJun28onehour;
% filter_datetime=(mergedurbandd.datetime>"2024-06-28 06:01:59");
% ufilter_datetime=(mergedurbandd.datetime>"2024-06-28 06:55:51" & mergedurbandd.datetime<"2024-06-28 06:56:47" );
% filter_ele = (mergedurbandd.U2I_Elevation > 30);
% filter_SNR = (mergedurbandd.U2I_SNR > 35);
% filter_GPS = (mergedurbandd.Master_sat_id <=32);
% filter_GLONASS = (mergedurbandd.Master_sat_id >32 & mergedurbandd.Master_sat_id <=82);
% filter_BDS = (mergedurbandd.Master_sat_id >82);
% mergedurbandd_sel = mergedurbandd(filter_ele & filter_SNR & filter_datetime & ~ufilter_datetime,:);
% tData_sorted = sortrows(mergedurbandd_sel,1);
% all_epoches = sort(unique(tData_sorted.datetime));
%     
% %---------------Calculate position and bound---------------%
% T = 0.01; % SAMPLE INTERVAL
% i=170;
% epoch = all_epoches(i);
% % select epoch
% filter_date= (tData_sorted.datetime==epoch);
% Xdata_org=tData_sorted(filter_date,:);
% % random select N rows
% num_sat_list = [];
% cal_time_list = [];
% for N = 3:size(Xdata_org,1)
%     N
%     numbers = 1:size(Xdata_org,1);
%     shuffled_numbers = numbers(randperm(length(numbers)));
%     indices = shuffled_numbers(1:N);
%     Xdata = Xdata_org(indices,:);
%     for jj=1:500 % Monte-Carlo simulation
%         % meas
%         meas = (Xdata.user2master_pseudorange - Xdata.ref2master_pseudorange) ...
%               - (Xdata.user2target_pseudorange - Xdata.ref2target_pseudorange); % (rho_s_r1 - rho_s_r2) - (rho_i_r1 - rho_i_r2)
%         meas = -meas;
%         % sat position
%         m_sv_pos = [Xdata.master_x,Xdata.master_y,Xdata.master_z];
%         s_sv_pos = [Xdata.target_x,Xdata.target_y,Xdata.target_z];
%         % other info.
%         init_state=zeros(3,1);
% 
%         % LS positioning
%         meas_std = ones(size(meas));
%         [eWLSSolution,prn_res,ErrorECEF,G,eDeltaPos,eDeltaPr] = YanFuncLib_Unity_tmp.WeightedLeastSquareDD(GTusr_xyz, init_state, meas, meas_std, m_sv_pos,s_sv_pos,ref_xyz);
%         % get scale list
%         S = inv(G'*G)*G';
%         S_enu = YanFuncLib_Unity_tmp.trans_solmat_ecef2enu(S,M_ecef2enu);
%         scale_list=S_enu(3,:); % related to z error (enu)   
%         % set definition domain of the position domain
%         x_scale=-30:T:30;
%         PHMI=1e-9;
%         [PL_pgo,PL_gaussian,fft_time_all]=YanFuncLib_Overbound_tmp.cal_PL(scale_list,x_scale,params_tsgo,params_pgo,PHMI,'ob_discrete');
%         num_sat_list(end+1) = size(Xdata,1);
%         cal_time_list(end+1) = fft_time_all;
%     end
% end
% 
% % %---------------summarize and plot computation time---------------%
% format long
% A = cal_time_list'; % array to be cluster
% B = num_sat_list';  % index of clusers
% % use excel to remove outliers, producing edit_AB
% A = edit_AB(:,2);
% B = edit_AB(:,1);
% uniqueValues = unique(B);
% meanValue_list=groupsummary(A,B,'mean');
% % plot computation time of PL
% figure
% scatter(B,A,'ko')
% hold on
% plot(uniqueValues,meanValue_list,'r-*');
% ylabel('Computation time (s)');
% xlabel('Number of measurements');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% save('Urban_PL_computation_time_MonteCarlo_20240715.mat')