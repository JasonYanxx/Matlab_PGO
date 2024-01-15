YanFun=Yan_functions;
seed=1234;
load('CHTI_overbounding.mat');
% load('ref_overbounding_correction.mat');
figure
for i=1:length(ele_start_list)
        ele_start=ele_start_list(i);
        % load Data
        [Xdata,x_lin,pdf_data]=YanFun.load_RefSPP('Data/cors_CHTI_Jan/mergedCHTIJan_exd.mat',ele_start,5);
%         [Xdata,x_lin,pdf_data]=YanFun.load_RefDD('Data/mnav_zmp1_jan_20240105/mergedRefJan.mat',ele_start,5);

        pdf_emp = ksdensity(Xdata,x_lin);
        cdf_emp=cumtrapz(pdf_emp);
        cdf_emp=cdf_emp*(x_lin(2)-x_lin(1));
    
        [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
        counts=length(x_lin);
        % Two-step Gaussian overbound (zero-mean)
        [mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFun.two_step_bound_zero(Xdata,x_lin);
        param_tsgo = std_tsgo;
        
        %------------- 20230110 -------------%
        if i==6 
            param_tsgo = std_tsgo*1.25;
            cdf_tsgo = normcdf(x_lin,0,param_tsgo);
        end
        if i==7
             param_tsgo = std_tsgo*1.35;
            cdf_tsgo = normcdf(x_lin,0,param_tsgo);
        end
        %------------- 20230110 -------------%
        
        % Principal Gaussian overbound (zero-mean)
        % retrive
        gmm_dist_raw=gmm_cells{i};
%         inflate_core=inflate_cells{1,i};
%         inflate_tail=inflate_cells{2,i};
%         thr=inflate_cells{3,i};
        inflate_core=1;
        inflate_tail=1;
        thr=0.7;
        gmm_dist_inflate=YanFun.inflate_GMM(gmm_dist_raw,inflate_core,inflate_tail); % inflate
        [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist_inflate,thr);
        
        close all
        % show pdf
        figure
        subplot(1,2,1)
        % plot(x_lin,pdf_data,'k','LineWidth',2);
        histogram(Xdata,'normalization','pdf')
        hold on
        plot(x_lin,pdf_emp,'k--','LineWidth',2);
        plot(x_lin,pdf_tsgo,'g','LineWidth',2);
        plot(x_lin,pdf_pgo,'b','LineWidth',2);
        xlabel('Error','FontSize',12);
        ylabel('PDF','FontSize',12);
        A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
        set(A,'FontSize',12)
        % show cdf
        subplot(1,2,2)
        plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
        hold on
        plot(x_lin,cdf_emp,'k--','LineWidth',2);
        plot(x_lin,cdf_tsgo,'g','LineWidth',2);
        plot(x_lin,cdf_pgo,'b','LineWidth',2);
        xlabel('Error','FontSize',12);
        ylabel('CDF','FontSize',12);
        A = legend('sample dist.','emp dist.','two-step','Principal Gaussian');
        set(A,'FontSize',12)
        
        % log scale cdf (left side)
%         subplot(4,3,i)
        figure;
        h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
        hold on
        h21=semilogy(x_lin,cdf_tsgo,'g','LineWidth',1);
        h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
        xlim([min(x_lin)*1.2,max(x_lin)*0.5])
        ylim([1e-5,1]);
        yticks([1e-5 1])
        if i==1 || i==4 || i==7 || i==10
            ylabel('CDF (log scale)');
        end
        if i==10 || i==11 || i==12
            xlabel('Error (m)');
        end
        title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
        set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
        grid on
        
        
        % log scale cdf (right side)
%         subplot(4,3,i)
        figure;
        h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 2);
        hold on
        h24=semilogy(x_lin,1-cdf_tsgo,'g','LineWidth',1);
        h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 2,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
        xlim([min(x_lin)*0.5,max(x_lin)*1.2])
        ylim([1e-5,1]);
        yticks([1e-5 1])
        yticklabels({'10^{-5}','10^{0}'})
        if i==1 || i==4 || i==7 || i==10
            ylabel('CCDF (log scale)');
        end
        if i==10 || i==11 || i==12
            xlabel('Error (m)');
        end
        title(['Elev.: ',num2str(ele_start),'\circ\sim',num2str(ele_start+5),'\circ'])
        set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
        grid on
        
        a=0;
end