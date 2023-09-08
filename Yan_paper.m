%% Paper-Overbounding compare: Fig.3, Fig.6
% YanFun=Yan_functions;
% seed=1234;
% % load Data
% % [Xdata,x_lin,pdf_data]=YanFun.load_RefDD();
% [Xdata,x_lin,pdf_data]=YanFun.load_UrbanDD();
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% counts=length(x_lin);
% 
% % Principal Gaussian overbound (zero-mean)
% % Type | Ele. | Inflate  | alpha |
% % Ref  | 30-35| 1,   1.15|  0.7  |
% % Ref  | 60-65| ?        |  0.7  |
% % Urban| 30-35| 1,   2.2 |  0.7  |
% % Urban| 30-80| 1.2, 2   |  0.9  |
% gmm_dist_raw=YanFun.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFun.inflate_GMM(gmm_dist_raw,1,2.2) 
% [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,0.7); 
% 
% % qq plot
% Xnorm=randn(1,length(Xdata));
% h=qqplot(Xdata,Xnorm);
% xlabel('Quantiles of error distribution (m)');
% ylabel('Standard normal quantile (m)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% grid on
% 
% % cdf
% figure;
% h1=plot(x_lin_ecdf,ecdf_data,'k','LineWidth',2);
% hold on
% % Two step Gaussian
% [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFun.two_step_bound(Xdata,x_lin);
% h21=plot(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'g','LineWidth',2);
% h24=plot(x_lin(params.idx+1:end),cdf_right_tsgo(params.idx+1:end),'g--','LineWidth',2);
% % Gaussian Pareto
% [params_gpo,pdf_gpo,cdf_gpo]=YanFun.Gaussian_Pareto_bound(Xdata,x_lin);
% h3=plot(x_lin,cdf_gpo,'r','LineWidth',2);
% h5=plot(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% xlabel('Error');
% ylabel('CDF');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h21,h24,h3,h5],'Sample dist.','Gaussian (Left)','Gaussian (Right)','Gaussian-Pareto','Principal Gaussian');
% set(A,'FontSize',13.5)
% grid on
% 
% % log scale cdf (left side)
% figure;
% h1=semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4);
% hold on
% h21=semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'g','LineWidth',2);
% h3=semilogy(x_lin,cdf_gpo,'r','LineWidth',2);
% h5=semilogy(x_lin,cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h21,h3,h5],'Sample dist.','Gaussian','Gaussian-Pareto','Principal Gaussian');
% set(A,'FontSize',13.5)
% grid on
% 
% 
% % log scale cdf (right side)
% figure;
% h1=semilogy(x_lin_ecdf,1-ecdf_data,'k+','LineWidth',1,'MarkerSize', 4);
% hold on
% h24=semilogy(x_lin(params.idx+1:end),1-cdf_right_tsgo(params.idx+1:end),'g','LineWidth',2);
% h3=semilogy(x_lin,1-cdf_gpo,'r','LineWidth',2);
% h5=semilogy(x_lin,1-cdf_pgo,'bd-','LineWidth',1,'MarkerSize', 4,'MarkerIndices',1:floor(length(x_lin)/100):length(x_lin));
% xlim([min(x_lin)*0.5,max(x_lin)*1.2])
% xlabel('Error (m)');
% ylabel('CCDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend([h1,h24,h3,h5],'Sample dist.','Gaussian','Gaussian-Pareto','Principal Gaussian');
% set(A,'FontSize',13.5)
% grid on

%% Paper-Urban DGNSS error against SNR and Ele: Fig.5
% load('Data/Urban_dd_0816/mergeurbandd.mat');
% figure;
% % ������ɫӳ��
% cmap = hot; % ʹ�� jet ɫͼ
% cmap = flipud(cmap);
% c = abs(mergedurbandd.doubledifferenced_pseudorange_error); % ������ά���� z ��Ϊ��ɫ
% c=log(c);
% % ���� 2D ɢ��ͼ����ʹ�õ���ά������ʾ��ɫ
% scatter(mergedurbandd.U2I_Elevation, mergedurbandd.U2I_SNR, 30, c, 'filled');
% colormap(cmap);
% colorbar;
% xlabel('Ele');
% ylabel('SNR');


%% VPL and VPE series, PL compuation time: Fig. 7a,b
YanFun=Yan_functions;
seed=1234;
% load GMM
[Xdata,x_lin_org,pdf_data]=YanFun.load_UrbanDD();
[ecdf_data, x_lin_ecdf] = ecdf(Xdata);
counts=length(x_lin_org);
% Principal Gaussian overbound (zero-mean)
gmm_dist_raw=YanFun.gene_GMM_EM_zeroMean(Xdata);
gmm_dist=YanFun.inflate_GMM(gmm_dist_raw,2,1.5); % inflate: 1.15; inflate: (2,1.5)
[params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin_org,gmm_dist,0.7);
% two-step Gaussian overbound (zero-mean)
[mean_tsgo, std_tsgo, pdf_tsgo, cdf_tsgo]=YanFun.two_step_bound_zero(Xdata,x_lin_org);
% expand the definition domian of the range domain error
lim=50;
Nsamples=100000;
AugCounts=floor((Nsamples-length(x_lin_org))/2);
delta_lin=abs(x_lin_org(1)-x_lin_org(2));
x_lin_exd_left= linspace(-lim, min(x_lin_org)-delta_lin, AugCounts);
x_lin_exd_right= linspace(max(x_lin_org)+delta_lin,lim, AugCounts);
x_lin_exd=[x_lin_exd_left x_lin_org x_lin_exd_right];
% obtain the excact value of pgo on the extended definition domain
[pdf_pgo_exd,~,~]=YanFun.two_piece_pdf(x_lin_exd,params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); 

% set transformation matrix: ecef to enu
% ���ߴ������ϵ��ռ�ֱ������ϵת�� https://www.lddgo.net/convert/coordinate-transform
p_ecef=[-2418235.676841056 , 5386096.899553243 , 2404950.408609563];
p_lbh=[114.1790017,22.29773881,3];
p.L=p_lbh(1);p.B=p_lbh(2);p.H=p_lbh(3);
p.Xp=p_ecef(1);p.Yp=p_ecef(2);p.Zp=p_ecef(3);
M=YanFun.matrix_ecef2enu(p);

% PL 
min_s=10000;
min_s_file='';
file_error='Data/Least_square_dd_urbandata/error_LSDD.csv';
error_data = readmatrix(file_error, 'NumHeaderLines', 1);
PL_pgo_list=zeros(length(error_data),1);
PL_gaussian_list=zeros(length(error_data),1);
cal_time_list=zeros(length(error_data),1);
num_sat_list=zeros(length(error_data),1);
xerr_list=zeros(length(error_data),1);
zerr_list=zeros(length(error_data),1);
gps_week=2238;
for i=1:length(error_data)
    % read file
%     % SPP file
%     gps_sec=error_data(i,1);
%     unix_sec= gps_week * 604800.0 + gps_sec + 315964800.0 + 19.0;
%     xy_error=error_data(i,2);
%     xyz_error=error_data(i,3);
    % DGNSS file
    unix_sec = error_data(i,1);
    xy_error=error_data(i,4);
    xyz_error=error_data(i,5);
    
    z_error=sqrt(xyz_error^2-xy_error^2);
    xerr_list(i)=xy_error;
    zerr_list(i)=z_error;
    
    % Open file by filename wildcard 
    folder = 'Data/Least_square_dd_urbandata/DD_S_matrix';
    field=num2str(unix_sec);
    wildcard = fullfile(folder, ['*' field '*']);
    fileList = dir(wildcard);
    if isempty(fileList)
        continue
    end
    filename = fullfile(folder, fileList(1).name);
    S_mat = load(filename);
    
    % skip invalid file
%     % for SPP
%     if size(S_mat,1)~=6 
%         continue
%     end
    % for DGNSS
    if size(S_mat,2)<3 
        continue
    end
    
    % use the positioning part of S matrix
    S_matp=S_mat(1:3,:);
    % agumentation
    S_matpa=zeros(size(S_matp,1)+1,size(S_matp,2)+1);
    S_matpa(1:size(S_matp,1),1:size(S_matp,2))=S_matp;
    S_matpa(end,end)=1;
    % transform
    S_matTrans=M*S_matpa;
    % use the core part
    S_matTransCore=S_matTrans(1:end-1,1:end-1);
  
%     scale_list=S_matTransCore(1,:); % related to x error (enu) min_s=0.001410985784173
%     scale_list=S_mat(2,:); % related to y error (ecef)  min_s=4.974492589715496e-05
    scale_list=S_mat(3,:); % related to z error (ecef)  min_s=2.313112008120455e-04
    num_sat_list(i)=length(scale_list);
    
%     % for debug
%     if min_s>min(abs(scale_list))
%         min_s=min(abs(scale_list));
%         min_s_file=field;
%     end
   
    % set definition domain of the position domain
    x_scale=-30:0.01:30;
    try
        [PL_pgo,PL_gaussian,fft_time_all]=YanFun.cal_PL(x_lin_exd,pdf_pgo_exd,std_tsgo,scale_list,x_scale,params_pgo);
        PL_pgo_list(i)=PL_pgo;
        PL_gaussian_list(i)=PL_gaussian;
        cal_time_list(i)=fft_time_all;
    catch exception
        disp('Error!');
    end
end

err_list=zerr_list;
% save PE and PL for Stanford chart plot
% save("Urban_PL2.mat","err_list","PL_pgo","PL_gaussian")

% plot times series of PL and PE
figure
yyaxis left
h1=plot(1:length(error_data),abs(err_list),'k-','linewidth',1.5);
hold on
h2=plot(1:length(error_data),abs(PL_gaussian_list),'g-','linewidth',1);
h3=plot(1:length(error_data),abs(PL_pgo_list),'b-','linewidth',1);
ylim([0 200])
ylabel('HPL (m)','FontSize',12);
yyaxis right
h4=plot(1:length(error_data),num_sat_list,'c:','linewidth',1);
ylim([0 55])
ylabel('Measurement counts');
xlabel('Time (s)');
xlim([0 1100])
ax = gca;
ax.YAxis(1).Color = 'black';
ax.YAxis(2).Color = 'black';
A = legend([h1,h2,h3,h4],'Error','Gaussian','Principal Gaussian','Counts');
set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
set(A,'FontSize',13.5)

% summarize computation time
format long
A = cal_time_list; % array to be cluster
B = num_sat_list;  % index of clusers
uniqueValues = unique(B);
meanValue_list=groupsummary(A,B,'mean');
% plot computation time of PL
figure
scatter(num_sat_list,cal_time_list,'ko')
hold on
plot(uniqueValues,meanValue_list,'r-*');
ylabel('Computation time (s)');
xlabel('Number of measurements');
set(gca, 'FontSize', 15,'FontName', 'Times New Roman');


%% paper-Stanford chart: Fig. 7c,d
% load('Urban_PL.mat')
% E=abs(VPE');
% % PL=abs(VPL_Gaussian');
% PL=abs(VPL_pgo');
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
%                 'Step',            3,...
%                 'Maximum',         300,...
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

%% Paper-alpha effects: Fig. 8
% YanFun=Yan_functions;
% seed=1234;
% % load Data
% [Xdata,x_lin,pdf_data]=YanFun.load_RefDD();
% [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
% counts=length(x_lin);
% 
% % Two step Gaussian
% [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=YanFun.two_step_bound(Xdata,x_lin);
% % Gaussian Pareto
% [params_gpo,pdf_gpo,cdf_gpo]=YanFun.Gaussian_Pareto_bound(Xdata,x_lin);
% 
% % Principal Gaussian overbound (zero-mean)
% % Type | Ele. | Inflate  | alpha |
% % Ref  | 30-35| 1,   1.15|  0.7  |
% gmm_dist_raw=YanFun.gene_GMM_EM_zeroMean(Xdata);
% gmm_dist=YanFun.inflate_GMM(gmm_dist_raw,1,1.15) 
% 
% % ָ����ɫӳ�����ƺ����鳤��
% % log scale cdf (left side)
% figure;
% semilogy(x_lin_ecdf,ecdf_data,'k+','LineWidth',1,'MarkerSize', 4,'DisplayName','Sample dist.');
% hold on
% semilogy(x_lin(1:params.idx),cdf_left_tsgo(1:params.idx),'g','LineWidth',2,'DisplayName','Gaussian');
% semilogy(x_lin,cdf_gpo,'r','LineWidth',2,'DisplayName','Gaussian-Pareto');
% mapName = 'jet';
% counts = 2;
% % ������ɫӳ��
% cmap = colormap(mapName);
% step=floor(length(cmap)/counts)+1;
% % ��ȡָ�����ȵ���ɫӳ��
% colorArray  = cmap(1:step:end, :);
% for i=1:counts
%     alpha=i*0.05+0.5;
%     [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,alpha); 
%     semilogy(x_lin,cdf_pgo,'LineWidth',1,'DisplayName',num2str(alpha),'color',colorArray(i,:));
% end
% xlim([min(x_lin)*1.2,max(x_lin)*0.5])
% xlabel('Error (m)');
% ylabel('CDF (log scale)');
% set(gca, 'FontSize', 15,'FontName', 'Times New Roman');
% A = legend();
% set(A,'FontSize',13.5)
% grid on

%% Bias effects and paired Principal Gaussian overbound: future work
% YanFun=Yan_functions;
% seed=1234;
% % load GMM
% [Xdata,x_lin,pdf_data,cdf_data,~]=YanFun.load_GMM_bias(seed);
% gmm_dist=YanFun.gene_GMM_EM_zeroMean(Xdata);
% pdf_emp=pdf(gmm_dist,x_lin')';
% cdf_emp=cdf(gmm_dist,x_lin')';
% 
% % Principal Gaussian overbound (zero-mean)
% [params_pgo, pdf_pgo, cdf_pgo]=YanFun.Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,0.7);
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
% gmm_dist_left=YanFun.gene_GMM_EM_zeroMean(Xleft_recon-mean(Xleft_recon));
% gmm_dist_left=YanFun.inflate_GMM(gmm_dist_left,1,1.1)
% [params_pgo_left, pdf_pgo_left, cdf_pgo_left]=YanFun.Principal_Gaussian_bound(Xleft_recon-mean(Xleft_recon),x_lin,gmm_dist_left,0.6);
% counts=length(x_lin);
% plot(x_lin(1:floor(counts/2))+mean(Xleft_recon),pdf_pgo_left(1:floor(counts/2))*length(Xdata)*0.2,'r','LineWidth',2);
% 
% figure;
% histogram(Xleft_recon,'BinWidth',0.2,'FaceColor','b');
% hold on
% Xleft_recon_center=Xleft_recon-Xmedian;
% histogram(Xleft_recon_center,'BinWidth',0.2,'FaceColor','g');
% plot(x_lin,pdf_pgo_left*length(Xdata)*0.2,'r','LineWidth',2);

function d = Display(e,s)
    if s > 0.01
        d = sprintf('%s(%.3f%%)',char(10),100*s);
    else
        d = sprintf('%s(%u epochs)',char(10),e);
    end
end

