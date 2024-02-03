%% addjust overbounding parameters (both pgo and tsgo) (be careful when using it)
% YanFun=Yan_functions;
% seed=1234;
% % load('CHTI_overbounding.mat');
% % load('ref_overbounding.mat');
% figure
% for i=1:length(ele_start_list)
%         ele_start=ele_start_list(i);
%         % load Data
% %         [Xdata,x_lin,pdf_data]=YanFun.load_RefSPP('Data/cors_CHTI_Jan/mergedCHTIJan_exd.mat',ele_start,5);
%         [Xdata,x_lin,pdf_data]=YanFun.load_RefDD('Data/mnav_zmp1_jan_20240105/mergedRefJan.mat',ele_start,5);
% 
%         pdf_emp = ksdensity(Xdata,x_lin);
%         cdf_emp=cumtrapz(pdf_emp);
%         cdf_emp=cdf_emp*(x_lin(2)-x_lin(1));
%     
%         [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
%         counts=length(x_lin);
%         % Two-step Gaussian overbound (zero-mean)
%         param_tsgo = tsgo_cells{i};
%         pdf_tsgo = normpdf(x_lin,0,param_tsgo);
%         cdf_tsgo = normcdf(x_lin,0,param_tsgo);
%         
%         % Principal Gaussian overbound (zero-mean)
%         % retrive
%         gmm_dist_raw=gmm_cells{i};
%         inflate_core=inflate_cells{1,i};
%         inflate_tail=inflate_cells{2,i};
%         thr=inflate_cells{3,i};
%         if length(inflate_core)==0
%             disp("no param")
%             inflate_core=1;
%             inflate_tail=1;
%             thr=0.7;
%         end
%         gmm_dist_inflate=YanFun.inflate_GMM(gmm_dist_raw,inflate_core,inflate_tail); % inflate
%         [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist_inflate,thr);
%         
%         close all
%         % show pdf
%         figure
%         subplot(2,2,1)
%         % plot(x_lin,pdf_data,'k','LineWidth',2);
%         histogram(Xdata,'normalization','pdf')
%         hold on
%         plot(x_lin,pdf_emp,'k--','LineWidth',2);
%         plot(x_lin,pdf_tsgo,'g','LineWidth',2);
%         plot(x_lin,pdf_pgo,'b','LineWidth',2);
%         xlabel('Error','FontSize',12);
%         ylabel('PDF','FontSize',12);
% 
%         % show cdf
%         subplot(2,2,2)
%         plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
%         hold on
%         plot(x_lin,cdf_emp,'k--','LineWidth',2);
%         plot(x_lin,cdf_tsgo,'g','LineWidth',2);
%         plot(x_lin,cdf_pgo,'b','LineWidth',2);
%         xlabel('Error','FontSize',12);
%         ylabel('CDF','FontSize',12);
%         A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
%         set(A,'FontSize',12)
%         
%         % log scale cdf (left side)
%         subplot(2,2,3)
%         h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
%         hold on
%         h21=semilogy(x_lin,cdf_tsgo,'g','LineWidth',1);
%         h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%         xlim([min(x_lin)*1.2,max(x_lin)*0.5])
%         ylim([1e-5,1]);
%         yticks([1e-5 1])
%         if i==1 || i==4 || i==7 || i==10
%             ylabel('CDF (log scale)');
%         end
%         if i==10 || i==11 || i==12
%             xlabel('Error (m)');
%         end
%         title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
%         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
%         grid on
%         
%         
%         % log scale cdf (right side)
%         subplot(2,2,4)
%         h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
%         hold on
%         h24=semilogy(x_lin,1-cdf_tsgo,'g','LineWidth',1);
%         h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%         xlim([min(x_lin)*0.5,max(x_lin)*1.2])
%         ylim([1e-5,1]);
%         yticks([1e-5 1])
%         yticklabels({'10^{-5}','10^{0}'})
%         if i==1 || i==4 || i==7 || i==10
%             ylabel('CCDF (log scale)');
%         end
%         if i==10 || i==11 || i==12
%             xlabel('Error (m)');
%         end
%         title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
%         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
%         grid on
%         
%         set_break_point_here=0;
%         % save corrected parameters
%         tsgo_cells{i}=param_tsgo;
%         gmm_cells{i}=gmm_dist_raw;
%         pgo_cells{i}=params_pgo;
%         inflate_cells{1,i}=inflate_core;
%         inflate_cells{2,i}=inflate_tail;
%         inflate_cells{3,i}=thr;
% end
% % save('CHTI_overbounding_correction',"gmm_cells","tsgo_cells","pgo_cells","ele_start_list","inflate_cells")
% % save('ref_overbounding_correction',"gmm_cells","tsgo_cells","pgo_cells","ele_start_list","inflate_cells")

%% visualize overbounding 
YanFun=Yan_functions;
seed=1234;
% load('CHTI_overbounding_correction.mat');
load('ref_overbounding_correction.mat');
figure
for i=1:min(12,length(ele_start_list))
        ele_start=ele_start_list(i);
        % load Data
%         [Xdata,x_lin,pdf_data]=YanFun.load_RefSPP('Data/cors_CHTI_Jan/mergedCHTIJan_exd.mat',ele_start,5);
        [Xdata,x_lin,pdf_data]=YanFun.load_RefDD('Data/mnav_zmp1_jan_20240105/mergedRefJan.mat',ele_start,5);

        pdf_emp = ksdensity(Xdata,x_lin);
        cdf_emp=cumtrapz(pdf_emp);
        cdf_emp=cdf_emp*(x_lin(2)-x_lin(1));
    
        [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
        counts=length(x_lin);
        % Two-step Gaussian overbound (zero-mean)
        param_tsgo = tsgo_cells{i};
        cdf_tsgo = normcdf(x_lin,0,param_tsgo);
        
        % Principal Gaussian overbound (zero-mean)
        % retrive
        gmm_dist_raw=gmm_cells{i};
        inflate_core=inflate_cells{1,i};
        inflate_tail=inflate_cells{2,i};
        thr=inflate_cells{3,i};
        gmm_dist_inflate=YanFun.inflate_GMM(gmm_dist_raw,inflate_core,inflate_tail); % inflate
        [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist_inflate,thr);

%         % log scale cdf (left side) 4x3
%         subplot(4,3,i)
%         h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
%         hold on
%         h21=semilogy(x_lin,cdf_tsgo,'g','LineWidth',1);
%         h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%         xlim([min(x_lin)*1.2,max(x_lin)*0.5])
%         ylim([1e-5,1]);
%         yticks([1e-5 1])
%         if i==1 || i==4 || i==7 || i==10
%             ylabel('CDF (log scale)');
%         end
%         if i==10 || i==11 || i==12
%             xlabel('Error (m)');
%         end
%         title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
%         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
%         grid on
        
        
%         % log scale cdf (right side) 4x3
%         subplot(4,3,i)
%         h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
%         hold on
%         h24=semilogy(x_lin,1-cdf_tsgo,'g','LineWidth',1);
%         h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%         xlim([min(x_lin)*0.5,max(x_lin)*1.2])
%         ylim([1e-5,1]);
%         yticks([1e-5 1])
%         yticklabels({'10^{-5}','10^{0}'})
%         if i==1 || i==4 || i==7 || i==10
%             ylabel('CCDF (log scale)');
%         end
%         if i==10 || i==11 || i==12
%             xlabel('Error (m)');
%         end
%         title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
%         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
%         grid on


        % log scale cdf (left side) 3x4 figure
        subplot(3,4,i)
        h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
        hold on
        h21=semilogy(x_lin,cdf_tsgo,'g','LineWidth',1);
        h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
        xlim([min(x_lin)*1.2,max(x_lin)*0.5])
        ylim([1e-5,1]);
        yticks([1e-5 1])
        if i==1 || i==5 || i==9
            ylabel('CDF (log scale)');
        end
        if i==9 || i==10 || i==11 || i==12
            xlabel('Error (m)');
        end
        title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
        set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
        grid on
        
%         % log scale cdf (right side) 3x4
%         subplot(3,4,i)
%         h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
%         hold on
%         h24=semilogy(x_lin,1-cdf_tsgo,'g','LineWidth',1);
%         h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
%         xlim([min(x_lin)*0.5,max(x_lin)*1.2])
%         ylim([1e-5,1]);
%         yticks([1e-5 1])
%         yticklabels({'10^{-5}','10^{0}'})
%         if i==1 || i==5 || i==9
%             ylabel('CCDF (log scale)');
%         end
%         if i==9 || i==10 || i==11 || i==12
%             xlabel('Error (m)');
%         end
%         title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
%         set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
%         grid on
end

%% check consistency in corrected parameters
% % load('ref_overbounding_correction.mat');
% for i=1:length(ele_start_list)
%     i
%     params_pgo=pgo_cells{i};
%     gmm_dist_raw=gmm_cells{i};
%     inflate_core=inflate_cells{1,i};
%     inflate_tail=inflate_cells{2,i};
%     if params_pgo.p1~=gmm_dist_raw.ComponentProportion(1)
%         disp('prob is not same')
%     end
%     if abs(params_pgo.sigma1-gmm_dist_raw.Sigma(:,:,1)*inflate_core)>0.1
%         disp('sigma 1 is not same')
%     end
%     if abs(params_pgo.sigma2-gmm_dist_raw.Sigma(:,:,2)*inflate_tail)>0.1
%         disp('sigma 2 is not same')
%     end
%     a=0;
% end


