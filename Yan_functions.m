function YanFun=Yan_functions
YanFun.load_UrbanDD=@load_UrbanDD;
YanFun.load_RefDD=@load_RefDD;
YanFun.extend_CHTI=@extend_CHTI;
YanFun.load_RefSPP=@load_RefSPP;
YanFun.load_GNSS=@load_GNSS;
YanFun.load_GMM=@load_GMM;
YanFun.load_GMM_bias=@load_GMM_bias;
YanFun.load_NIG=@load_NIG;
YanFun.two_step_bound=@two_step_bound;
YanFun.two_step_bound_zero=@two_step_bound_zero;
YanFun.Gaussian_Pareto_bound=@Gaussian_Pareto_bound;
YanFun.total_Gaussian_bound=@total_Gaussian_bound;
YanFun.Principal_Gaussian_bound=@Principal_Gaussian_bound;
YanFun.two_piece_pdf=@two_piece_pdf;
YanFun.stable_bound=@stable_bound;
YanFun.distConv_org=@distConv_org;
YanFun.distSelfConv=@distSelfConv;
YanFun.get_conv=@get_conv;
YanFun.compareConvOverbound=@compareConvOverbound;
YanFun.cal_PL=@cal_PL;
YanFun.cal_PL_ex=@cal_PL_ex;
YanFun.FDE_Gaussian=@FDE_Gaussian;
YanFun.FDE_BayesGMM_seperate=@FDE_BayesGMM_seperate;
YanFun.FDE_BayesGMM_union=@FDE_BayesGMM_union;
YanFun.geneStateGMM=@geneStateGMM;
YanFun.FDE_mc_compare=@FDE_mc_compare;
YanFun.T1transpp_bound=@T1transpp_bound;
YanFun.gen_s1_s2=@gen_s1_s2;
YanFun.compare_twoside_bound=@compare_twoside_bound;
YanFun.gene_GMM_EM_zeroMean=@gene_GMM_EM_zeroMean;
YanFun.gene_GMM_EM_zeroMean_loose=@gene_GMM_EM_zeroMean_loose;
YanFun.cal_omega=@cal_omega;
YanFun.binary_search=@binary_search;
YanFun.inflate_GMM=@inflate_GMM;
YanFun.customrand=@customrand;
YanFun.nig_pdf=@nig_pdf;
YanFun.matrix_ecef2enu=@matrix_ecef2enu;
end


%% Generate Urban data
function [Xdata,x_lin,pdf_data]=load_UrbanDD(file,ele_start,ele_step)
    
    if nargin==0
        load('Data/urban_dd_0816/mergeurbandd.mat');
        % select: ele(30~35)   
        filter_ele=(mergedurbandd.U2I_Elevation>=30 & mergedurbandd.U2I_Elevation<=35); 
%         % select: ele(60~65)
%         filter_ele=(mergedRefOnemonth.U2I_Elevation>=60 & mergedRefOnemonth.U2I_Elevation<=65); 
    else
        load(file);
        filter_ele=(mergedurbandd.U2I_Elevation>=ele_start & mergedurbandd.U2I_Elevation<=ele_start+ele_step); 
    end
    
    % select: err(-15~15)
    filter_err=(mergedurbandd.doubledifferenced_pseudorange_error>=-15 & mergedurbandd.doubledifferenced_pseudorange_error<=15); 
    filter_SNR=(mergedurbandd.U2I_SNR>=40); 
    
    Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele & filter_SNR);
    Nsamples=length(Xdata);
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    cdf_data=cumtrapz(pdf_data);
    cdf_data=cdf_data*(x_lin(2)-x_lin(1));
end

%% Generate DGNSS CORS data
function [Xdata,x_lin,pdf_data]=load_RefDD(file, ele_start,ele_step)
    if nargin==0
        load('Data/urban_dd_0816/mergeurbandd.mat');
        mergedRefOnemonth=mergedRefJan;
        % select: ele(30~35)   
        filter_ele=(mergedRefOnemonth.U2I_Elevation>=30 & mergedRefOnemonth.U2I_Elevation<=35); 
        %     % select: ele(60~65)
    %     filter_ele=(mergedRefOnemonth.U2I_Elevation>=60 & mergedRefOnemonth.U2I_Elevation<=65); 
    else
        load(file);
        mergedRefOnemonth=mergedRefJan;
        filter_ele=(mergedRefOnemonth.U2I_Elevation>=ele_start & mergedRefOnemonth.U2I_Elevation<=ele_start+ele_step); 
    end
    % select: err(-15~15)
    filter_err=(mergedRefOnemonth.doubledifferenced_pseudorange_error>=-15 & mergedRefOnemonth.doubledifferenced_pseudorange_error<=15); 

    Xdata=mergedRefOnemonth.doubledifferenced_pseudorange_error(filter_ele & filter_err);
    
    Nsamples=length(Xdata);
    if Nsamples<15000
        Nsamples=15000;
    end
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
end

%% Generate SPP CORS data
function [Xdata,x_lin,pdf_data]=load_RefSPP(file, ele_start,ele_step)
    if nargin==0
        load('Data/cors_CHTI_Jan/mergedCHTIJan_exd.mat');
        mergedRefOnemonth=mergedCHTIJan_exd;
        % select: ele(30~35)   
        filter_ele=(mergedRefOnemonth.ele>=30*3.1415/180 & mergedRefOnemonth.ele<=35*3.1415/180); 
    else
        load(file);
        mergedRefOnemonth=mergedCHTIJan_exd;
        filter_ele=(mergedRefOnemonth.ele>=ele_start*3.1415/180 & mergedRefOnemonth.ele<=(ele_start+ele_step)*3.1415/180); 
    end
  
    % select: err(-15~15)
    filter_err=(mergedRefOnemonth.residual>=-15 & mergedRefOnemonth.residual<=15); 
    
    % selec: date
    % 2023-01-25:1674604800~1674691170
    % 2023-01-28:1674864000~1674950370
    filter_day25=(mergedRefOnemonth.gps_time>=1674604800 & mergedRefOnemonth.gps_time<=1674691170);
    filter_day28=(mergedRefOnemonth.gps_time>=1674864000 & mergedRefOnemonth.gps_time<=1674950370);
    filter_dayuse= ~(filter_day25 | filter_day28);
    
    mergedRefSel = mergedRefOnemonth(filter_dayuse & filter_ele & filter_err,:);
    Xdata=mergedRefOnemonth.residual(filter_dayuse & filter_ele & filter_err);
    
    Nsamples=length(Xdata);
    if Nsamples<15000
        Nsamples=15000;
    end
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
end


function obs_exd_all=extend_CHTI(on_data)
CLIGHT=3e8;
recPos=[-43.735473315,-176.617116049,75.6896];% CHTI: lat lon alt
all_epochs = sort(unique(on_data.gps_time));
for i=1:length(all_epochs)
    epoch_cur=all_epochs(i);
    obs_cur = on_data(on_data.gps_time==epoch_cur,:);
    obs_sel=obs_cur(obs_cur.ele>=15*3.1415/180,:);
    num_sel = height(obs_sel);
    obs_remain=obs_cur(obs_cur.ele<15*3.1415/180,:);
    num_remain=height(obs_remain);
    % estimate receier clock bias with GTxyz
    warr=[];
    for ii=1:num_sel
        w=1/obs_sel.sigma(ii)^2;
        warr(1,ii)=w;
    end
    warr=warr/sum(warr);
    rec_dt = warr*(obs_sel.cor_pseudo-obs_sel.range)/CLIGHT;

    % calculate residual
    res = obs_sel.cor_pseudo-obs_sel.range-rec_dt*CLIGHT;
    
    % extend 
    obs_sel_exd = addvars(obs_sel, rec_dt*ones(num_sel,1), res,...
        'NewVariableNames',{'rec_dt', 'residual'});
    obs_remain_exd = addvars(obs_remain, NaN(num_remain,1), NaN(num_remain,1),...
        'NewVariableNames',{'rec_dt', 'residual'});

    % save all obs_cur_exd
    if i==1
        obs_exd_all=vertcat(obs_sel_exd, obs_remain_exd);
    else
        tmp = vertcat(obs_sel_exd, obs_remain_exd);
        obs_exd_all = vertcat(obs_exd_all, tmp);
    end
end

end

function [Xdata,x_lin,pdf_data,cdf_data]=load_GNSS()
    load('./sample_data.mat')
    Xdata=sample_data;
    
    Nsamples=length(Xdata);
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    [ecdf_data, x_lin_ecdf] = ecdf(Xdata);
end

function [Xdata,x_lin,pdf_data,cdf_data,dist]=load_GMM(seed)
    % p1=0.9774;
    % p2=1-p1;
    % mu1=0;
    % mu2=0;
    % sigma1=0.1342^2;
    % sigma2=0.4399^2;
    
%     p1=0.9;
%     p2=1-p1;
%     mu1=0;
%     mu2=0;
%     sigma1=0.2^2;
%     sigma2=1^2;
    
    p1=0.8;
    p2=1-p1;
    mu1=0;
    mu2=0;
    sigma1=0.02^2;
    sigma2=0.06^2;

    Nsamples=10001;
    dist = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
    rng(seed);
    Xdata = random(dist, Nsamples);
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    cdf_data=cumtrapz(pdf_data);
    cdf_data=cdf_data*(x_lin(2)-x_lin(1));
end

function [Xdata,x_lin,pdf_data,cdf_data,dist]=load_GMM_bias(seed)

    p1=0.8;
    p2=1-p1;
    mu1=-2;
    mu2=0.5;
    sigma1=1.3^2;
    sigma2=1.7^2;

    Nsamples=10001;
    dist = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
    rng(seed);
    Xdata = random(dist, Nsamples);
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    cdf_data=cumtrapz(pdf_data);
    cdf_data=cdf_data*(x_lin(2)-x_lin(1));
end


function [Xdata,x_lin,pdf_data,cdf_data]=load_NIG()  
    Nsamples=10001; % number of random numbers to generate
    interval = [-7, 7]; % interval over which pdf is defined
    M = 1; % constant M for acceptance-rejection method
    Xdata = customrand(@nig_pdf, interval, Nsamples, M);
    lim=max(-min(Xdata),max(Xdata));
    x_lin = linspace(-lim, lim, Nsamples);
    pdf_data = ksdensity(Xdata,x_lin);
    cdf_data=cumtrapz(pdf_data);
    cdf_data=cdf_data*(x_lin(2)-x_lin(1));
end

%% overbound
function [mean_overbound, std_overbound,pdf_overbound,cdf_overbound]=two_step_bound_zero(Xdata,x_lin)
    Nbins = 100;
    NstepsCdf = 1000;
    epsilon = 0.0025;
    % Right side overbound
    [mean_overbound, std_overbound, epsilon_achieved, intervals]=gaussian_overbound(Xdata, epsilon, Nbins, NstepsCdf);
    mean_overbound=0;
    if isempty(x_lin)
        pdf_overbound=0;
        cdf_overbound=0;
    else
        pdf_overbound=normpdf(x_lin,mean_overbound,std_overbound);
        cdf_overbound=normcdf(x_lin,mean_overbound,std_overbound);
    end
end

function [params,pdf_left,pdf_right,cdf_left,cdf_right]=two_step_bound(Xdata,x_lin)
    Nbins = 100;
    NstepsCdf = 1000;
    epsilon = 0.0025;
    % Right side overbound
    [mean_overbound_r, std_overbound_r, epsilon_r, intervals_r]=gaussian_overbound(Xdata, epsilon, Nbins, NstepsCdf);
    % Left side overbound
    [mean_overbound_l, std_overbound_l, epsilon_l, intervals_l]=gaussian_overbound(-Xdata, epsilon, Nbins, NstepsCdf);
    mean_overbound_l=-mean_overbound_l;
    
    mean_overbound=max(abs(mean_overbound_r),abs(mean_overbound_l));
    std_overbound=max(std_overbound_r,std_overbound_l);
    
    params.mean_overbound_r=mean_overbound;
    params.std_overbound_r=std_overbound;
    params.mean_overbound_l=-mean_overbound;
    params.std_overbound_l=std_overbound;
    
    Xmedian=median(Xdata);
    idx=ceil(length(x_lin)/2); % 0: original paper
    params.idx=idx;
%     idx = binary_search(x_lin, Xmedian); % median: possible improvement
    pdf_left=normpdf(x_lin,mean_overbound_l,std_overbound_l);
    pdf_right=normpdf(x_lin,mean_overbound_r,std_overbound_r);
    cdf_left=normcdf(x_lin,mean_overbound_l,std_overbound_l);
    cdf_right=normcdf(x_lin,mean_overbound_r,std_overbound_r);

end

function [params,pdf_overbound,cdf_overbound]=Gaussian_Pareto_bound(Xdata,x_lin)
    % right-tail overbound
    [params.thr_R,params.theta_R,params.xi_R,params.scale_R]=gp_tail_overbound(Xdata);
    % left-tail overbound
    [params.thr_L,params.theta_L,params.xi_L,params.scale_L]=gp_tail_overbound(-Xdata);
    params.thr_L=-params.thr_L;
    % zero-mean Gaussian core overbound - Two-step method
    Nbins = 100;
    NstepsCdf = 1000;
    epsilon = 0.0025;
    % note: use all data instead of data in the interval
    [params.mean_core, params.std_core, epsilon_achieved, intervals]=gaussian_overbound(Xdata, epsilon, Nbins, NstepsCdf);
    params.mean_core=0;
    
    if isempty(x_lin)
        pdf_overbound=0;
        cdf_overbound=0;
    else
        cdf_overbound=GaussPareto_cdf(x_lin,params);
        pdf_overbound = diff(cdf_overbound) ./ (x_lin(2)-x_lin(1));
        pdf_overbound(end+1) = pdf_overbound(end);
    end

end

function [mean_overbound, std_overbound,pdf_overbound,cdf_overbound]=total_Gaussian_bound(Xdata,x_lin,gmm_dist)
    if isempty(gmm_dist)
%         options = statset('TolFun', 1e-6, 'MaxIter', 10000);
%         gmm_dist = fitgmdist(Xdata, 2, 'Options', options);
        gmm_dist=gene_GMM_EM_zeroMean(Xdata);
    end
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;

    mut=p1*mu1+p2*mu2;
    sigmat=p1*sigma1+(mut-mu1)*(mut-mu1)+p2*sigma2+(mut-mu2)*(mut-mu2);

    mean_overbound=mut;
    std_overbound=sqrt(sigmat);
    if isempty(x_lin)
        pdf_overbound=0;
        cdf_overbound=0;
    else
        pdf_overbound=normpdf(x_lin,mean_overbound,std_overbound);
        cdf_overbound=normcdf(x_lin,mean_overbound,std_overbound);
    end
end

function [params,pdf_overbound,cdf_overbound]=Principal_Gaussian_bound(Xdata,x_lin,gmm_dist,thr)
    if isempty(gmm_dist)
%         options = statset('TolFun', 1e-6, 'MaxIter', 10000);
%         gmm_dist = fitgmdist(Xdata, 2, 'Options', options);
        gmm_dist=gene_GMM_EM_zeroMean(Xdata);
    end
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;

    % ����bayes' method ����s1, s2 �ֲ�
    if isempty(x_lin)
        lim=max(-min(Xdata),max(Xdata));
        x= linspace(-lim, lim, length(Xdata));
    else
        x=x_lin;
    end
    Nsamples=length(x);
    s1_list=zeros(1,Nsamples);
    s2_list=zeros(1,Nsamples);
    for j=1:Nsamples
        [s1,s2]=cal_omega(x(j),mu1,sigma1,p1,mu2,sigma2,p2);
        s1_list(j)=s1;
        s2_list(j)=s2;
    end

    % ���� s1 �� s2 �ķ�λ��
    s1_half=s1_list(1:round(Nsamples/2));
    idx_xL1p=binary_search(s1_half, max(min(s1_list),thr*max(s1_list)));
    xL1p=x(idx_xL1p);
    s1_half_right=s1_list(round(Nsamples/2)+1:Nsamples);
    s1_half_right=flip(s1_half_right);
    idx_xR1p=binary_search(s1_half_right, max(min(s1_list),thr*max(s1_list)));
    idx_xR1p = Nsamples - idx_xR1p;
    xR1p=x(idx_xR1p);

    s2_half=s2_list(round(Nsamples/2)+1:Nsamples);
    idx_xR2p=binary_search(s2_half, max(min(s2_list),thr*max(s2_list)));
    idx_xR2p=idx_xR2p+round(Nsamples/2);
    xR2p=x(idx_xR2p);
    s2_half_left = s2_list(1:round(Nsamples/2));
    s2_half_left = flip(s2_half_left);
    idx_xL2p=binary_search(s2_half_left, max(min(s2_list),thr*max(s2_list)));
    idx_xL2p=round(Nsamples/2)-idx_xL2p;
    xL2p=x(idx_xL2p);

    
%     [pdf_piece_fun]=three_piece_pdf(x,gmm_dist,xL2p,xR2p,xL1p,xR1p);
%     [cdf_piece_fun_join]=three_piece_cdf(x,gmm_dist,xL2p,xR2p,idx_xL2p,idx_xR2p,...,
%                                                xL1p,xR1p,idx_xL1p,idx_xR1p);
    
    [pdf_piece_fun,k,cc]=two_piece_pdf(x,gmm_dist,xL2p,xR2p); % two piece
    [cdf_piece_fun_join]=two_piece_cdf(x,gmm_dist,xL2p,xR2p,idx_xL2p,idx_xR2p);
    
    params.k=k;
    params.cc=cc;
    params.sigma1=sigma1;% variance
    params.sigma2=sigma2;% variance
    params.p1=p1;
    params.gmm_dist=gmm_dist;
    params.xL2p=xL2p;
    params.xR2p=xR2p;
    params.s1_list=s1_list;
    params.s2_list=s2_list;

    pdf_overbound=pdf_piece_fun;
    cdf_overbound=cdf_piece_fun_join;
%     % numerical way
%     cdf_overbound=cumtrapz(pdf_overbound);
%     cdf_overbound=cdf_overbound*(x(2)-x(1));
   
end

function [alpha_overbound, gama_overbound,pdf_overbound,cdf_overbound]=stable_bound(Xdata,x_lin)
    Nbins = 100;
    NstepsCdf = 1000;
    epsilon = 0.0025;
    [alpha_overbound,gama_overbound,epsilon_achieved,intervals]=stable_overbound(Xdata, epsilon, Nbins, NstepsCdf);
    sas_dist=makedist('Stable','alpha',alpha_overbound,'beta',0,'gam',gama_overbound,'delta',0);
    
    if isempty(x_lin)
        pdf_overbound=0;
        cdf_overbound=0;
    else
        pdf_overbound =pdf(sas_dist,x_lin);
        cdf_overbound =cdf(sas_dist,x_lin);
    end
end

function [T1trans_pdf,cdf_T1transpp_right,cdf_T1transpp_left]=T1transpp_bound(Xdata,x_lin,gmm_dist)
    if isempty(gmm_dist)
%         options = statset('TolFun', 1e-6, 'MaxIter', 10000);
%         gmm_dist = fitgmdist(Xdata, 2, 'Options', options);
        gmm_dist=gene_GMM_EM_zeroMean(Xdata);
    end
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    
    if isempty(x_lin)
        lim=max(-min(Xdata),max(Xdata));
        x_lin= linspace(-lim, lim, length(Xdata));
    end
    Nsamples=length(x_lin);
    
    T1trans_pdf=zeros(1,Nsamples);
    for i=1:Nsamples
        y=x_lin(1,i);
        [s1,s2]=cal_omega(y',mu1,sigma1,p1,mu2,sigma2,p2);
        [T1]=cal_T(y,mu1,sigma1,s1,mu2,sigma2,s2);
    %     T1transpp(1,i)=p1^max(s1-s2,0)*p2^max(s2-s1,0)*normpdf(y, s1*mu1+s2*mu2, s1*sqrt(sigma1)+s2*sqrt(sigma2));
        T1trans_pdf(1,i)=normpdf(y, s1*mu1+s2*mu2, s1*sqrt(sigma1)+s2*sqrt(sigma2));
    end


    % ���Ҳ࿪ʼ���ۻ��ֲ�
    T1transpp_inverse = flip(T1trans_pdf);
    cdf_T1transpp_right = cumtrapz(T1transpp_inverse);
    % cdf_T1transpp_right = cdf_T1transpp_right / max(cdf_T1transpp_right);% normalization
    cdf_T1transpp_right=cdf_T1transpp_right*(x_lin(2)-x_lin(1));
    cdf_T1transpp_right=flip(cdf_T1transpp_right);

    % ����࿪ʼ���ۻ��ֲ�
    cdf_T1transpp_left = cumtrapz(T1trans_pdf);
    % cdf_T1transpp_left = cdf_T1transpp_left / max(cdf_T1transpp_left);% normalization
    cdf_T1transpp_left=cdf_T1transpp_left*(x_lin(2)-x_lin(1));

end

function [s1_list,s2_list]=gen_s1_s2(x_lin,Xdata,gmm_dist,add_mu,ax)
    mu1=gmm_dist.mu(1)+add_mu;
    mu2=gmm_dist.mu(2)+add_mu;
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    % ����bayes' method ����s1, s2 �ֲ�
    x=x_lin;
    Nsamples=length(x);
    s1_list=zeros(1,Nsamples);
    s2_list=zeros(1,Nsamples);
    for j=1:Nsamples
        [s1,s2]=cal_omega(x(j),mu1,sigma1,p1,mu2,sigma2,p2);
        s1_list(j)=s1;
        s2_list(j)=s2;
    end
    histogram(ax,Xdata,'normalization','pdf')
    hold on
    pdf_data=p1*normpdf(x,mu1,sqrt(sigma1))+p2*normpdf(x,mu2,sqrt(sigma2));
    plot(ax,x,pdf_data,'k','LineWidth',2);
    plot(ax,x,p1*normpdf(x,mu1,sqrt(sigma1)),'r','LineWidth',2);
    plot(ax,x,p2*normpdf(x,mu2,sqrt(sigma2)),'b','LineWidth',2);
    plot(ax,x,s1_list,'r--','LineWidth',2);
    plot(ax,x,s2_list,'b--','LineWidth',2);
    xline(ax,quantile(Xdata,0.05/2));
    xline(ax,quantile(Xdata,1-0.05/2));
    xline(ax,median(Xdata));
end

%% Convolution
% convolution of two distribution 
function [pdf_conv,x_conv,conv_t]=distConv_org(x1,x2,pdf_dist1,pdf_dist2,method)
% https://ww2.mathworks.cn/matlabcentral/answers/1440944-request-for-help-computing-convolution-of-random-variables-via-fft
% define output length
N1 = length(pdf_dist1);       
N2 = length(pdf_dist2);  
N=N1+N2-1;
% define delta_x (x1 and x2 should have the same delta_x)
dx = x1(2) - x1(1);

% convolution
if method=="fft"
    tic;
    fft_pdf1=fft(pdf_dist1,N);
    fft_pdf2=fft(pdf_dist2,N);
    pdf_fftconv = ifft(fft_pdf1 .* fft_pdf2);
    conv_t=toc;
    pdf_conv=pdf_fftconv*dx; % multiple dx
elseif method=="direct"
    tic;
    pdf_conv=conv(pdf_dist1,pdf_dist2)*dx;
    conv_t=toc;    
end

min_x=min(x1)+min(x2);
max_x=max(x1)+max(x2);
%     x_fftconv=min_x:dx:max_x; % bug: not enough number
x_conv=linspace(min_x,max_x,N);
end

% self-convolution (multiple times)
function [pdf_conv,ts_all]=distSelfConv(x,pdf_dist,num_conv,method)
    ts_all=0;
    [pdf_conv,x_conv,conv_t]=distConv_org(x,x,pdf_dist,pdf_dist,method);
    pdf_conv=interp1(x_conv,pdf_conv,x,...
           'linear','extrap');
    ts_all=ts_all+conv_t;
    for i=1:num_conv-1
        [pdf_conv,x_conv,conv_t]=distConv_org(x,x,pdf_conv,pdf_dist,method);
        pdf_conv=interp1(x_conv,pdf_conv,x,...
           'linear','extrap');
        ts_all=ts_all+conv_t;
    end
end

function [yc,tt]=get_conv(x,y1,y2)
    x = x';
    xc = 2*x;
    tic;
    yc = conv(y1,y2)*(x(2)-x(1));
    tt=toc;
    yc = interp1(xc,yc(1:2:end),x,...
       'linear','extrap');
end

function compareConvOverbound(x,pdf_ob,pdf_data,num_conv)
    for i=1:num_conv
        [pdf_data,~]=distSelfConv(x,pdf_data,i,"fft");
        [pdf_ob,~]=distSelfConv(x,pdf_ob,i,"fft");
        cdf_data = cumtrapz(pdf_data)*(x(2)-x(1));
        cdf_ob = cumtrapz(pdf_ob)*(x(2)-x(1));
        h=figure;
        plot(x,cdf_data,'b','LineWidth',2);
        hold on
        plot(x,cdf_ob,'r','LineWidth',2);
        plot(x,sign(cdf_ob-cdf_data),'m','LineWidth',2);
        waitfor(h);
    end
end

%% Protection level
function [PL_pgo,PL_gaussian,fft_time_all]=cal_PL(scale_list,x_scale,std_tsgo,params_pgo,PHMI)
    if scale_list(1)==0
        return
    end
    fft_time_all=0;
    [func_conv,~,~]=two_piece_pdf(x_scale/scale_list(1),params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); % ����ʽ���
    x_conv=x_scale;
    
    coeff=abs(1/scale_list(1));
%     figure;plot(x_conv,func_conv); hold on
    for i=2:length(scale_list)
        s=scale_list(i);
        if s==0
            error('s=0');
        end

        [func_scale,~,~]=two_piece_pdf(x_scale/s,params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); % ����ʽ���
%         func_conv=conv(func_conv,func_scale)*0.01; % convolution
        [func_conv,x_conv,fft_time]=distConv_org(x_conv,x_scale,func_conv,func_scale,"fft"); % fft
        coeff=coeff*abs(1/s);
        fft_time_all=fft_time_all+fft_time;
%         plot(x_conv,func_conv);
    end
    pdf_obp=coeff*func_conv;
    cdf_obp=cumtrapz(pdf_obp);
    cdf_obp=cdf_obp*0.01;
%     figure; 
%     plot(x_conv,cdf_obp);

    xaxi_list=x_conv;
    cum_p=0;
    PL_pgo=999; % cannot solve PL_pgo
    for i=1:length(xaxi_list)
        if cum_p>sum(pdf_obp)*PHMI/2
            PL_pgo=xaxi_list(i);
            break
        end
        cum_p=cum_p+pdf_obp(i);
    end
    if PL_pgo==999
        error("PL_pgo cannot be solved");
    else
%         disp(PL_pgo)
        aa=0;
    end

    % Gaussian PL
    if std_tsgo~=[]
        std_position_tsgo=sum(abs(scale_list))*std_tsgo;
        PL_gaussian=norminv(1e-9/2,0,std_position_tsgo);
        disp(PL_gaussian)
    else
        PL_gaussian=999;
    end
end

%% Protection level - different nominal model
function [PL_pgo,PL_gaussian,fft_time_all]=cal_PL_ex(scale_list,x_scale,tsgo_current_cells,pgo_current_cells,PHMI)
    if scale_list(1)==0
        return
    end
    fft_time_all=0;
    params_pgo = pgo_current_cells{1};
    [func_conv,~,~]=two_piece_pdf(x_scale/scale_list(1),params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); % ����ʽ���
    x_conv=x_scale;
    
    coeff=abs(1/scale_list(1));
%     figure;plot(x_conv,func_conv); hold on
    for i=2:length(scale_list)
        s=scale_list(i);
        if s==0
            error('s=0');
        end
        params_pgo = pgo_current_cells{i};
        [func_scale,~,~]=two_piece_pdf(x_scale/s,params_pgo.gmm_dist,params_pgo.xL2p,params_pgo.xR2p); % ����ʽ���
%         func_conv=conv(func_conv,func_scale)*0.01; % convolution
        [func_conv,x_conv,fft_time]=distConv_org(x_conv,x_scale,func_conv,func_scale,"fft"); % fft
        coeff=coeff*abs(1/s);
        fft_time_all=fft_time_all+fft_time;
%         plot(x_conv,func_conv);
    end
    pdf_obp=coeff*func_conv;
    cdf_obp=cumtrapz(pdf_obp);
    cdf_obp=cdf_obp*0.01;
%     figure; 
%     plot(x_conv,cdf_obp);

    xaxi_list=x_conv;
    cum_p=0;
    PL_pgo=999; % cannot solve PL_pgo
    for i=1:length(xaxi_list)
        if cum_p>sum(pdf_obp)*PHMI/2
            PL_pgo=xaxi_list(i);
            break
        end
        cum_p=cum_p+pdf_obp(i);
    end
    if PL_pgo==999
        error("PL_pgo cannot be solved");
    else
%         disp(PL_pgo)
        aa=0;
    end

%     % Gaussian PL
%     if std_tsgo~=[]
%         std_position_tsgo=sum(abs(scale_list))*std_tsgo;
%         PL_gaussian=norminv(1e-9/2,0,std_position_tsgo);
%         disp(PL_gaussian)
%     else
%         PL_gaussian=999;
%     end
    PL_gaussian=999;
end

%% FDE
function [gmm_state]=geneStateGMM(gmm_dist,num)
    % ����state error ��GMM (zero-mean)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    p_group=[p1;p2];
    s_group=[sigma1;sigma2];
        
    [P{1:num}] = ndgrid(p_group); % ʹ��ndgrid����������ѭ��
    [S{1:num}] = ndgrid(s_group);
    prob_c = P{1};
    sigma_c = S{1};
    for i = 2:num
        prob_c = bsxfun(@times, prob_c, P{i});
        sigma_c = sigma_c + S{i};
    end
    prob_list = prob_c(:)'; % ��prob_c��sigma_c���ӳ�һά����
    sigma_list = sigma_c(:)';

    comps=length(sigma_list);
    sigma_all=num2cell(sigma_list);
    gmm_state = gmdistribution(zeros(comps,1), cat(3, sigma_all{:}), prob_list); % ����state error ��GMM�ֲ�
end

function [FA_o,MD_o]=FDE_Gaussian(alpha,seed,num,gmm_dist,bias,sigma)
    % ���ɹ۲����ݣ��в��ÿ��6���۲⣬һ��10000�� (false alarm)
    N=10000;
    T_o_arr=zeros(N,1);
    rng(seed);
    rk_all = random(gmm_dist, num*N);
    for i=1:N
        rk=rk_all((i-1)*num+1:i*num);
        T_o_arr(i)=rk'*rk/sigma;
    end
    FA_o=sum(sum(T_o_arr>chi2inv(1-alpha,num-1)))/N; % �Ƿ��ȥDOF
    
    % ÿ��6���۲⣨ĳ���۲�ע��һ����bias����һ��10000�� (miss detection)
    T_o_arr2=zeros(N,1);
    rng(seed);
    rk_all = random(gmm_dist, num*N);
    for i=1:N
        rk=rk_all((i-1)*num+1:i*num);
        rk(1)=bias;
        T_o_arr2(i)=rk'*rk/sigma;
    end
    MD_o=1-sum(sum(T_o_arr2>chi2inv(1-alpha,num-1)))/N; % �Ƿ��ȥDOF
end

function [FA_o,MD_o]=FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,method)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    % ���ɹ۲����ݣ��в��ÿ��6���۲⣬һ��10000�� (false alarm)
    N=10000;
    T_o_arr_max=zeros(N,1);
    T_o_arr_sum=zeros(N,1);
    rng(seed);
    rk_all = random(gmm_dist, num*N);
    for i=1:N
        rk=rk_all((i-1)*num+1:i*num);       
        % ʹ��Bayes method (seperate approach) ����ͳ����
        T1_list=[];
        for j=1:num
            [s1,s2]=cal_omega(rk(j),mu1,sigma1,p1,mu2,sigma2,p2);
            [T1]=cal_T(rk(j),mu1,sigma1,s1,mu2,sigma2,s2);
            T1_list(j)=T1^2;
        end
        T_o_arr_max(i)=max(T1_list);
        T_o_arr_sum(i)=sum(T1_list);
    end
    if method == "sum"
        FA_o=sum(sum(T_o_arr_sum>chi2inv(1-alpha,num-1)))/N; % �Ƿ��ȥDOF
    elseif method == "max"
        FA_o=sum(sum(T_o_arr_max>chi2inv(1-alpha,1)))/(N*num);
    end
    
    % ÿ��6���۲⣨ĳ���۲�ע��һ����bias����һ��10000�� (miss detection)
    T_o_arr_max2=zeros(N,1);
    T_o_arr_sum2=zeros(N,1);
    rng(seed);
    rk_all = random(gmm_dist, num*N);
    for i=1:N
        rk=rk_all((i-1)*num+1:i*num);
        rk(1)=bias;
        % ʹ��Bayes method (seperate approach) ����ͳ����
        T1_list=[];
        for j=1:num
            [s1,s2]=cal_omega(rk(j),mu1,sigma1,p1,mu2,sigma2,p2);
            [T1]=cal_T(rk(j),mu1,sigma1,s1,mu2,sigma2,s2);
            T1_list(j)=T1^2;
        end
        T_o_arr_max2(i)=max(T1_list);
        T_o_arr_sum2(i)=sum(T1_list);
    end
    if method == "sum"
        MD_o=1-sum(sum(T_o_arr_sum2>chi2inv(1-alpha,num-1)))/N; % �Ƿ��ȥDOF
    elseif method == "max"
        MD_o=1-sum(sum(T_o_arr_max2>chi2inv(1-alpha,1)))/(N*num);
    end
end

function [FA_arr,MD_arr,var_arr]=FDE_mc_compare(alpha,seed,num)
    N=10;
    FA_arr=zeros(N,4);
    MD_arr=zeros(N,4);
    var_arr=zeros(N,1);
    for i=1:N
        %�������ֲ�
        % varing sigma2
        % p1=0.98;
        % p2=1-p1;
        % mu1=0;
        % mu2=0;
        % sigma1=0.1^2;
        % sigma2=(0.2+i*(2-0.2)/N)^2;

        % varing p1
        p1=0.5+(i-1)*(1-0.5)/N;
        p2=1-p1;
        mu1=0;
        mu2=0;
        sigma1=0.1^2;
        sigma2=0.5^2;
        
        gmm_dist = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
        rng(seed);
        Xdata = random(gmm_dist, 100001);
        bias=quantile(Xdata,0.001/2);
        
         % Two-step Gaussian overbound (zero-mean)
        [mean_tsgo, std_tsgo, ~, ~]=two_step_bound_zero(Xdata,[]);
        % Total Gaussian overbound
        [mean_tgo, std_tgo, ~, ~]=total_Gaussian_bound(Xdata,[],gmm_dist);

        [FA_Gaussian,MD_Gaussian]=FDE_Gaussian(alpha,seed,num,gmm_dist,bias,std_tsgo^2);
        [FA_tGaussian,MD_tGaussian]=FDE_Gaussian(alpha,seed,num,gmm_dist,bias,std_tgo^2);
        [FA_Bayes_max,MD_Bayes_max]=FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,"max");
        [FA_Bayes_sum,MD_Bayes_sum]=FDE_BayesGMM_seperate(alpha,seed,num,gmm_dist,bias,"sum");
        
        FA_arr(i,:)=[FA_Gaussian,FA_tGaussian,FA_Bayes_max,FA_Bayes_sum];
        MD_arr(i,:)=[MD_Gaussian,MD_tGaussian,MD_Bayes_max,MD_Bayes_sum];
        var_arr(i)=p1;
    end
    
    % visualization
    figure
    subplot(1,2,1)
    plot(var_arr,FA_arr(:,1),'LineWidth',2)
    hold on
    plot(var_arr,FA_arr(:,2),'LineWidth',2)
    plot(var_arr,FA_arr(:,3),'LineWidth',2)
    plot(var_arr,FA_arr(:,4),'LineWidth',2)
    A = legend('Two-step','TotalGaussian','Bayes max','Bayes sum');
    set(A,'FontSize',12)
    xlabel('var','FontSize', 14)
    title('False alarm rate','FontSize', 14)
    ylim([-0.1,1.1])

    subplot(1,2,2)
    plot(var_arr,MD_arr(:,1),'LineWidth',2)
    hold on
    plot(var_arr,MD_arr(:,2),'LineWidth',2)
    plot(var_arr,MD_arr(:,3),'LineWidth',2)
    plot(var_arr,MD_arr(:,4),'LineWidth',2)
    A = legend('Two-step','TotalGaussian','Bayes max','Bayes sum');
    set(A,'FontSize',12)
    xlabel('var','FontSize', 14)
    title('Miss detection rate','FontSize', 14)
    ylim([-0.1,1.1])
end

%% others
function compare_twoside_bound(Xdata,x_lin,gmm_dist)
    if isempty(gmm_dist)
%         options = statset('TolFun', 1e-6, 'MaxIter', 10000);
%         gmm_dist = fitgmdist(Xdata, 2, 'Options', options);
        gmm_dist=gene_GMM_EM_zeroMean(Xdata);
    end
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    
    if isempty(x_lin)
        lim=max(-min(Xdata),max(Xdata));
        x= linspace(-lim, lim, length(Xdata));
    else
        x=x_lin;
    end
    Nsamples=length(x);
    
    % total mean & variance
    mut=p1*mu1+p2*mu2;
    sigmat=p1*sigma1+(mut-mu1)*(mut-mu1)+p2*sigma2+(mut-mu2)*(mut-mu2); 
    % T1 pp
    [T1transpp,cdf_T1transpp_right,cdf_T1transpp_left]=T1transpp_bound(Xdata,x,gmm_dist);
    % Two-step method
    [params,pdf_left_tsgo,pdf_right_tsgo,cdf_left_tsgo,cdf_right_tsgo]=two_step_bound(Xdata,x);
    mean_overbound_r=params.mean_overbound_r;
    std_overbound_r=params.std_overbound_r;
    mean_overbound_l=params.mean_overbound_l;
    std_overbound_l=params.std_overbound_l;
    
    % Right side overbound
    subplot(2,2,1);
    histogram(Xdata,'normalization','pdf')
    xlabel('error','FontSize', 14)
    ylabel('pdf','FontSize',14)
    title('Right side','FontSize', 14)
    hold on
    y = pdf(gmm_dist, x'); % using matlab function
    plot(x,y,'r','LineWidth',2);
    y_ob_r = normpdf(x,mean_overbound_r,std_overbound_r);
    plot(x,y_ob_r,'g','LineWidth',2); 
    plot(x,T1transpp,'b','LineWidth',2);
    yt = normpdf(x, mut, sqrt(sigmat));
    plot(x,yt,'m','LineWidth',2);
    A = legend('sample hist','sample dist.','two-step bounding dist.','T1transpp','TotalGaussian');
    set(A,'FontSize',12)

    subplot(2,2,2);
    y = 1-cdf(gmm_dist, x'); % using matlab function
    plot(x,y,'r','LineWidth',2);
    xlabel('error','FontSize', 14)
    ylabel('cdf','FontSize',14)
    title('Right side','FontSize', 14)
    hold
    y_ob_r = 1-normcdf(x,mean_overbound_r,std_overbound_r);
    plot(x,y_ob_r,'g','LineWidth',2); 
    plot(x,cdf_T1transpp_right,'b','LineWidth',2); 
    yt = 1-normcdf(x, mut, sqrt(sigmat));
    plot(x,yt,'m','LineWidth',2);
    A = legend('sample dist.','two-step bounding dist.','T1transpp','TotalGaussian');
    set(A,'FontSize',12)

    % Left side overbound 
    subplot(2,2,3);
    histogram(Xdata,'normalization','pdf')
    xlabel('error','FontSize', 14)
    ylabel('pdf','FontSize',14)
    title('Left side','FontSize', 14)
    hold on
    y = pdf(gmm_dist, x'); % using matlab function
    plot(x,y,'r','LineWidth',2);
    y_ob_l = normpdf(x,mean_overbound_l,std_overbound_l);
    plot(x,y_ob_l,'g','LineWidth',2); 
    plot(x,T1transpp,'b','LineWidth',2);
    yt = normpdf(x, mut, sqrt(sigmat));
    plot(x,yt,'m','LineWidth',2);
    A = legend('sample hist','sample dist.','two-step bounding dist.','T1transpp','TotalGaussian');
    set(A,'FontSize',12)

    subplot(2,2,4);
    y = cdf(gmm_dist, x'); % using matlab function
    plot(x,y,'r','LineWidth',2);
    xlabel('error','FontSize', 14)
    ylabel('cdf','FontSize',14)
    title('Left side','FontSize', 14)
    hold
    y_ob_l = normcdf(x,mean_overbound_l,std_overbound_l);
    plot(x,y_ob_l,'g','LineWidth',2); 
    plot(x,cdf_T1transpp_left,'b','LineWidth',2); 
    yt = normcdf(x, mut, sqrt(sigmat));
    plot(x,yt,'m','LineWidth',2);
    A = legend('sample dist.','two-step bounding dist.','T1transpp','TotalGaussian');
    set(A,'FontSize',12)


    disp('overbound mean    overbound sigma')
       [mean_overbound_l std_overbound_l;
        mean_overbound_r std_overbound_r]

end

%% utility functions
function X = customrand(y_pdf, interval, N, M)
% acceptance-rejection method to generate samples
    a = interval(1);
    b = interval(2);
    X = zeros(1, N);
    count = 0;
    while count < N
        x = a + (b - a).*rand(1, N);
        u = rand(1, N);
        idx = u <= y_pdf(x)./M;
        X(count+1:count+sum(idx)) = x(idx);
        count = count + sum(idx);
    end
    X = X(1:N);
    X=X';
end

function y = nig_pdf(data)
    % This is a simple implementation of the PDF of the NIG distribution.
    % x is the variable, alpha, beta, delta and mu are parameters of the NIG distribution.
    
    % below is the setting in "Rife, J., Pullen, S., & Pervan, B. (2004).
    % Core Overbounding and its Implications for LAAS Integrity. Proceedings 
    % of the 17th International Technical Meeting of the Satellite Division of 
    % The Institute of Navigation (ION GNSS 2004), 2810�C2821." with M=1
    mu=0;
    alpha=0.65;
    delta=0.65;
    beta=0;
    
    y=zeros(size(data));
    for i=1:size(data,2)
        x=data(i);
        y(i) = nig_function(x,alpha,beta,delta,mu);
    end
end

function f = nig_function(x,alpha,beta,delta,mu)
    gama=sqrt(alpha^2-beta^2);
    f = alpha*delta*exp(beta*(x-mu)+delta*gama)*besselk(1,alpha*sqrt(delta^2+(x-mu)^2))/(pi*sqrt(delta^2+(x-mu)^2));
end

function y=GaussPareto_cdf(data,params)
    y=zeros(size(data));
    for i=1:size(data,2)
        x=data(i);
        y(i) = GaussPareto_cdf_func(x,params);
    end
end

function [y]=GaussPareto_cdf_func(x,params)
    mean_core=params.mean_core;
    std_core=params.std_core;
    thr_L=params.thr_L;
    xi_L=params.xi_L;
    scale_L=params.scale_L;
    theta_L=params.theta_L;
    thr_R=params.thr_R;
    xi_R=params.xi_R;
    scale_R=params.scale_R;
    theta_R=params.theta_R;
    
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

function [threshold,theta,xi,scale]=gp_tail_overbound(xData)
    xData=sort(xData);
    Nsamples=length(xData);
    % adjust threhold
    threshold_min=quantile(xData,1-0.1);
    threshold_max=quantile(xData,1-250/Nsamples);
    thr_range=threshold_min:0.001:threshold_max;
    candidate_list=zeros(length(thr_range),5);
    for i=1:length(thr_range)
        threshold=thr_range(i);
        fit_xData = xData(xData >= threshold);
        fit_xData=fit_xData-threshold; % move to center
        % mle 
        params = mle(fit_xData, 'distribution', 'GeneralizedPareto');
        theta=0; % location
        xi=params(1); % shape
        scale=params(2); % scale
        pd=makedist('gp','k',xi,'sigma',scale,'theta',theta);
        x_ext=icdf(pd,1-1e-7);
        candidate_list(i,:)=[threshold theta xi scale x_ext];
    end

    % select params yeild the largest x_ext
    [~, index] = max(candidate_list(:,5));
    maxRow = candidate_list(index,:);
    [threshold,theta,xi,scale,x_ext]=deal(maxRow(1), maxRow(2), maxRow(3), maxRow(4), maxRow(5));
end

function [omega1,omega2]=cal_omega(y,mu1,sigma1,pi_1,mu2,sigma2,pi_2)
sum_p =pi_1* mvnpdf(y',mu1',sigma1)+ pi_2* mvnpdf(y',mu2',sigma2);
omega1 = pi_1* mvnpdf(y',mu1',sigma1) / sum_p;
omega2 = pi_2* mvnpdf(y',mu2',sigma2) / sum_p;
end

function [T1]=cal_T(y,mu1,sigma1,s1,mu2,sigma2,s2)
    weight_sig =s1 * inv(sqrtm(sigma1)) + s2 * inv(sqrtm(sigma2));
    weight_mu = s1 * mu1 + s2 * mu2;
    T1=weight_sig * (y - weight_mu);
end

function [pdf_piece_fun,k,cc]=two_piece_pdf(x,gmm_dist,xL2p,xR2p)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    % ����ֶκ�����pdf -- two piece
    Nsamples=length(x);
    pdf_piece_fun=zeros(1,Nsamples);
    k=p1*normcdf(xL2p,mu1,sqrt(sigma1))/(p2*normcdf(xL2p,mu2,sqrt(sigma2)));
    cc=p2*(normcdf(xL2p,mu2,sqrt(sigma2))-0.5)/xL2p;
    for j=1:Nsamples
        if x(j)<xL2p
            pp=(1+k)*p2*normpdf(x(j),mu2,sqrt(sigma2));
        elseif x(j)<0
            pp=p1*normpdf(x(j),mu1,sqrt(sigma1))+cc;
        elseif x(j)<xR2p
            pp=p1*normpdf(x(j),mu1,sqrt(sigma1))+cc;
        else
            pp=(1+k)*p2*normpdf(x(j),mu2,sqrt(sigma2));
        end
        pdf_piece_fun(j)=pp;
    end
end

function [pdf_piece_fun]=three_piece_pdf(x,gmm_dist,xL2p,xR2p,xL1p,xR1p)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    % ����ֶκ�����pdf -- three piece
    Nsamples=length(x);
    pdf_piece_fun=zeros(1,Nsamples);
    k=p1*normcdf(xL2p,mu1,sqrt(sigma1))/(p2*normcdf(xL2p,mu2,sqrt(sigma2)));
    rme_left=p2*normcdf(xL2p,mu2,sqrt(sigma2))+p2*normpdf(xL1p,mu2,sqrt(sigma2))*(xL1p-xL2p)-0.5*p2;
    rme_right=p2*normpdf(xR1p,mu2,sqrt(sigma2))*(xR2p-xR1p)...
                -p2*normcdf(xR2p,mu2,sqrt(sigma2))...
                +0.5*p2;
    for j=1:Nsamples
        if x(j)<xL2p
            pp=(1+k)*p2*normpdf(x(j),mu2,sqrt(sigma2));
        elseif x(j)<xL1p
            pp=p1*normpdf(x(j),mu1,sqrt(sigma1))+p2*normpdf(xL1p,mu2,sqrt(sigma2));
        elseif x(j)<0
            pp=p1*normpdf(x(j),mu1,sqrt(sigma1))-rme_left/(-xL1p);
        elseif x(j)<xR1p
            pp=p1*normpdf(x(j),mu1,sqrt(sigma1))-rme_right/xR1p;
        elseif x(j)<xR2p
            pp=p1*normpdf(x(j),mu1,sqrt(sigma1))+p2*normpdf(xR1p,mu2,sqrt(sigma2));
        else
            pp=(1+k)*p2*normpdf(x(j),mu2,sqrt(sigma2));
        end
        pdf_piece_fun(j)=pp;
    end
end

function [cdf_piece_fun_join]=two_piece_cdf(x,gmm_dist,xL2p,xR2p,idx_xL2p,idx_xR2p)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    % ����ֶκ�����pdf -- two piece
    Nsamples=length(x);
    k=p1*normcdf(xL2p,mu1,sqrt(sigma1))/(p2*normcdf(xL2p,mu2,sqrt(sigma2)));
    cc=p2*(normcdf(xL2p,mu2,sqrt(sigma2))-0.5)/xL2p;
    cdf_piece_left_2 = (1+k)*p2*normcdf(x(1:idx_xL2p-1), mu2, sqrt(sigma2));

    cdf_piece_core_left_2 = p1*normcdf(x(idx_xL2p:round(Nsamples/2)-1),mu1,sqrt(sigma1))...
                    -p1*normcdf(x(idx_xL2p-1),mu1,sqrt(sigma1))...
                    +(x(idx_xL2p:round(Nsamples/2)-1)-x(idx_xL2p-1))*cc;
    cdf_piece_core_left_2 = cdf_piece_core_left_2 + cdf_piece_left_2(size(cdf_piece_left_2,2));

    cdf_piece_core_right_2 = p1*normcdf(x(round(Nsamples/2):idx_xR2p-1),mu1,sqrt(sigma1))...
                    -p1*normcdf(x(round(Nsamples/2)-1),mu1,sqrt(sigma1))...
                    +(x(round(Nsamples/2):idx_xR2p-1)-x(round(Nsamples/2)))*cc;
    cdf_piece_core_right_2 = cdf_piece_core_right_2 + cdf_piece_core_left_2(size(cdf_piece_core_left_2,2));


    cdf_piece_right_2 =(1+k)*p2*normcdf(x(idx_xR2p:Nsamples),mu2,sqrt(sigma2))...
                        -(1+k)*p2*normcdf(x(idx_xR2p-1),mu2,sqrt(sigma2));
    cdf_piece_right_2=cdf_piece_right_2+cdf_piece_core_right_2(size(cdf_piece_core_right_2,2));
    cdf_piece_fun_join=[cdf_piece_left_2 cdf_piece_core_left_2 cdf_piece_core_right_2 cdf_piece_right_2];
end

function [cdf_piece_fun_join]=three_piece_cdf(x,gmm_dist,xL2p,xR2p,idx_xL2p,idx_xR2p,...,
                                               xL1p,xR1p,idx_xL1p,idx_xR1p)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    % ����ֶκ�����pdf -- two piece
    Nsamples=length(x);
    k=p1*normcdf(xL2p,mu1,sqrt(sigma1))/(p2*normcdf(xL2p,mu2,sqrt(sigma2)));
    rme_left=p2*normcdf(xL2p,mu2,sqrt(sigma2))+p2*normpdf(xL1p,mu2,sqrt(sigma2))*(xL1p-xL2p)-0.5*p2;
    rme_right=p2*normpdf(xR1p,mu2,sqrt(sigma2))*(xR2p-xR1p)...
                -p2*normcdf(xR2p,mu2,sqrt(sigma2))...
                +0.5*p2;
    cdf_piece_left_tail = (1+k)*p2*normcdf(x(1:idx_xL2p-1), mu2, sqrt(sigma2));

    cdf_piece_left = p1*normcdf(x(idx_xL2p:idx_xL1p-1),mu1,sqrt(sigma1))...
                    -p1*normcdf(x(idx_xL2p-1),mu1,sqrt(sigma1))...
                    +(x(idx_xL2p:idx_xL1p-1)-x(idx_xL2p-1))*p2*normpdf(xL1p,mu2, sqrt(sigma2));
    cdf_piece_left = cdf_piece_left + cdf_piece_left_tail(size(cdf_piece_left_tail,2));

    cdf_piece_core_left = p1*normcdf(x(idx_xL1p:round(Nsamples/2)-1),mu1,sqrt(sigma1))...
                         - p1*normcdf(x(idx_xL1p-1),mu1,sqrt(sigma1))...
                         - rme_left*(x(idx_xL1p:round(Nsamples/2)-1)-x(idx_xL1p-1))/(-xL1p);
    cdf_piece_core_left = cdf_piece_core_left + cdf_piece_left(size(cdf_piece_left,2));

    cdf_piece_core_right =  p1*normcdf(x(round(Nsamples/2):idx_xR1p-1),mu1,sqrt(sigma1))...
                           -p1*normcdf(x(round(Nsamples/2)-1),mu1,sqrt(sigma1))...
                           -rme_right*(x(round(Nsamples/2):idx_xR1p-1)-x(round(Nsamples/2)))/xR1p;
    cdf_piece_core_right = cdf_piece_core_right+cdf_piece_core_left(size(cdf_piece_core_left,2));       

    cdf_piece_right = p1*normcdf(x(idx_xR1p:idx_xR2p-1),mu1,sqrt(sigma1))...
                    -p1*normcdf(x(idx_xR1p),mu1,sqrt(sigma1))...
                    +(x(idx_xR1p:idx_xR2p-1)-x(idx_xR1p-1))*p2*normpdf(xR1p,mu2, sqrt(sigma2));
    cdf_piece_right = cdf_piece_right + cdf_piece_core_right(size(cdf_piece_core_right,2));


    cdf_piece_right_tail =(1+k)*p2*normcdf(x(idx_xR2p:Nsamples),mu2,sqrt(sigma2))...
                        -(1+k)*p2*normcdf(x(idx_xR2p-1),mu2,sqrt(sigma2));

    cdf_piece_right_tail=cdf_piece_right_tail+cdf_piece_right(size(cdf_piece_right,2));
    cdf_piece_fun_join=[cdf_piece_left_tail cdf_piece_left cdf_piece_core_left cdf_piece_core_right cdf_piece_right cdf_piece_right_tail];

end

function idx = binary_search(arr, target)
    left = 1;
    right = length(arr);
    
    while left <= right
        mid = floor((left + right) / 2);
        if abs(arr(mid) - target)<1e-1 % watch out: 1e-3
            idx = mid;
            return;
        elseif arr(mid) < target
            left = mid + 1;
        else
            right = mid - 1;
        end
    end
    idx = -1;  % ���δ�ҵ�Ŀ��ֵ������-1
end


function [gmm_dist]=gene_GMM_EM_zeroMean(Xdata)
    samples =Xdata;
    D=size(samples,2);
    N=size(samples,1);

    K=2;%��K����̬�ֲ����
    Pi=ones(1,K)/K;
%     Pi=[0.9 0.1];
    Miu={0;0};
    Sigma2=cell(K,1);
%     %% %K��ֵ����ȷ����ֵ
%     [idx,center]=kmeans(samples,K);
%     for i=1:K
% %         miu0=center(i,:);
%         sigma0=var(samples(find(idx==i),:));
% %         Miu{i,1}=miu0;
%         Sigma2{i,1}=sigma0;
%     end
   %%  ȷ����ֵ
    if K==2
        Sigma2{1,1}=var(samples(abs(samples)<quantile(samples,1-0.05))); %core area
        Sigma2{2,1}=var(samples(abs(samples)>quantile(samples,1-0.05))); %tail area
    end

    beta=inf;
    likelihood_function_value=0;
    record=[];
    %% %EM�㷨
    while(1)
        %% %E��
        gama=zeros(N,K); % membership weight
        samples_pd=zeros(N,K);
        for j=1:K
            samples_pd(:,j)=normpdf(samples,Miu{j,1},sqrt(Sigma2{j,1}));
        end
        for i=1:N
            for j=1:K
                gama(i,j)=Pi(j)*samples_pd(i,j)/(Pi*samples_pd(i,:)');
            end
        end

        likelihood_function_value_old=likelihood_function_value;
        likelihood_function_value=sum(log(sum(samples_pd.*repmat(Pi,N,1),1)));
        record=[record,likelihood_function_value];
        beta=abs(likelihood_function_value-likelihood_function_value_old);    
        if beta<1e-6
%             plot(1:length(record),record)
            break
        end
        %% %M��
        Nk=sum(gama,1);
        for j=1:K
            Miu{j,1}=zeros(1,D);
            Sigma2{j,1}=zeros(D,D);
%             for i=1:N
%                 Miu{j,1}=Miu{j,1}+gama(i,j)*samples(i,:)/Nk(j);
%             end 
            for i=1:N
                Sigma2{j,1}=Sigma2{j,1}+(gama(i,j)*(samples(i,:)-Miu{j,1})'*(samples(i,:)-Miu{j,1}))/Nk(j);
            end
        end
        Pi=Nk/N;
    end
    
    % exchange components: Sort in acscending order of sigma
    [~, idx] =sort(cell2mat(Sigma2)', 'ascend');
    Pi=Pi(idx);
    Miu = Miu(idx);
    Sigma2 = Sigma2(idx);
    gmm_dist = gmdistribution(vertcat(Miu{:}), cat(3, Sigma2{:}), Pi);
end

function [gmm_dist]=gene_GMM_EM_zeroMean_loose(Xdata)

    samples =Xdata;

    D=size(samples,2);
    N=size(samples,1);

    K=2;%��K����̬�ֲ����
    Pi=ones(1,K)/K;
%     Pi=[0.9 0.1];
    Miu={0;0};
    Sigma2=cell(K,1);
%     %% %K��ֵ����ȷ����ֵ
%     [idx,center]=kmeans(samples,K);
%     for i=1:K
% %         miu0=center(i,:);
%         sigma0=var(samples(find(idx==i),:));
% %         Miu{i,1}=miu0;
%         Sigma2{i,1}=sigma0;
%     end
   %%  ȷ����ֵ
    if K==2
        Sigma2{1,1}=var(samples); 
        Sigma2{2,1}=var(samples)*1.5;
    end

    beta=inf;
    likelihood_function_value=0;
    record=[];
    %% %EM�㷨
    while(1)
        %% %E��
        gama=zeros(N,K); % membership weight
        samples_pd=zeros(N,K);
        for j=1:K
            samples_pd(:,j)=normpdf(samples,Miu{j,1},sqrt(Sigma2{j,1}));
        end
        for i=1:N
            for j=1:K
                gama(i,j)=Pi(j)*samples_pd(i,j)/(Pi*samples_pd(i,:)');
            end
        end

        likelihood_function_value_old=likelihood_function_value;
        likelihood_function_value=sum(log(sum(samples_pd.*repmat(Pi,N,1),1)));
        record=[record,likelihood_function_value];
        beta=abs(likelihood_function_value-likelihood_function_value_old);    
        if beta<1e-6
%             plot(1:length(record),record)
            break
        end
        %% %M��
        Nk=sum(gama,1);
        for j=1:K
            Miu{j,1}=zeros(1,D);
            Sigma2{j,1}=zeros(D,D);
%             for i=1:N
%                 Miu{j,1}=Miu{j,1}+gama(i,j)*samples(i,:)/Nk(j);
%             end 
            for i=1:N
                Sigma2{j,1}=Sigma2{j,1}+(gama(i,j)*(samples(i,:)-Miu{j,1})'*(samples(i,:)-Miu{j,1}))/Nk(j);
            end
        end
        Pi=Nk/N;
    end
    
    % exchange components: Sort in acscending order of sigma
    [~, idx] =sort(cell2mat(Sigma2)', 'ascend');
    Pi=Pi(idx);
    Miu = Miu(idx);
    Sigma2 = Sigma2(idx);
    gmm_dist = gmdistribution(vertcat(Miu{:}), cat(3, Sigma2{:}), Pi);
end

function [gmm_dist]=gene_GMM_EM_zeroMean_tailall(Xdata)
    samples =Xdata;
    D=size(samples,2);
    N=size(samples,1);

    K=2;%��K����̬�ֲ����
    Pi=ones(1,K)/K;
%     Pi=[0.9 0.1];
    Miu={0;0};
    Sigma2=cell(K,1);
%     %% %K��ֵ����ȷ����ֵ
%     [idx,center]=kmeans(samples,K);
%     for i=1:K
% %         miu0=center(i,:);
%         sigma0=var(samples(find(idx==i),:));
% %         Miu{i,1}=miu0;
%         Sigma2{i,1}=sigma0;
%     end
   %%  ȷ����ֵ
    if K==2
        Sigma2{1,1}=var(samples(abs(samples)<quantile(samples,1-0.05))); %core area
        Sigma2{2,1}=var(samples); % all area including the tail area
    end

    beta=inf;
    likelihood_function_value=0;
    record=[];
    %% %EM�㷨
    while(1)
        %% %E��
        gama=zeros(N,K); % membership weight
        samples_pd=zeros(N,K);
        for j=1:K
            samples_pd(:,j)=normpdf(samples,Miu{j,1},sqrt(Sigma2{j,1}));
        end
        for i=1:N
            for j=1:K
                gama(i,j)=Pi(j)*samples_pd(i,j)/(Pi*samples_pd(i,:)');
            end
        end

        likelihood_function_value_old=likelihood_function_value;
        likelihood_function_value=sum(log(sum(samples_pd.*repmat(Pi,N,1),1)));
        record=[record,likelihood_function_value];
        beta=abs(likelihood_function_value-likelihood_function_value_old);    
        if beta<1e-6
%             plot(1:length(record),record)
            break
        end
        %% %M��
        Nk=sum(gama,1);
        for j=1:K
            Miu{j,1}=zeros(1,D);
            Sigma2{j,1}=zeros(D,D);
%             for i=1:N
%                 Miu{j,1}=Miu{j,1}+gama(i,j)*samples(i,:)/Nk(j);
%             end 
            for i=1:N
                Sigma2{j,1}=Sigma2{j,1}+(gama(i,j)*(samples(i,:)-Miu{j,1})'*(samples(i,:)-Miu{j,1}))/Nk(j);
            end
        end
        Pi=Nk/N;
    end
    
    % exchange components: Sort in acscending order of sigma
    [~, idx] =sort(cell2mat(Sigma2)', 'ascend');
    Pi=Pi(idx);
    Miu = Miu(idx);
    Sigma2 = Sigma2(idx);
    gmm_dist = gmdistribution(vertcat(Miu{:}), cat(3, Sigma2{:}), Pi);
end


function [inflate_dist]=inflate_GMM(gmm_dist,inflation_factor1,inflation_factor2)
    mu1=gmm_dist.mu(1);
    mu2=gmm_dist.mu(2);
    sigma1=gmm_dist.Sigma(1);
    sigma2=gmm_dist.Sigma(2);
    p1=gmm_dist.ComponentProportion(1);
    p2=1-p1;
    
    sigma1=sigma1*inflation_factor1;
    sigma2=sigma2*inflation_factor2;
       
    inflate_dist = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
end

function [gmm_dist]=gene_GMM_EM(Xdata)

    samples =Xdata;

    D=size(samples,2);
    N=size(samples,1);

    K=2;%��K����̬�ֲ����
    Pi=ones(1,K)/K;
    Pi=[0.9 0.1];
    Miu={0;0};
    Sigma2=cell(K,1);
    %% %K��ֵ����ȷ����ֵ
    [idx,center]=kmeans(samples,K);
    miu0=center(1,:);
    for i=1:K
        sigma0=var(samples(find(idx==i),:));
        Miu{i,1}=miu0;
        Sigma2{i,1}=sigma0;
    end
    

    beta=inf;
    likelihood_function_value=0;
    record=[];
    %% %EM�㷨
    while(1)
        %% %E��
        gama=zeros(N,K); % membership weight
        samples_pd=zeros(N,K);
        for j=1:K
            samples_pd(:,j)=normpdf(samples,Miu{j,1},sqrt(Sigma2{j,1}));
        end
        for i=1:N
            for j=1:K
                gama(i,j)=Pi(j)*samples_pd(i,j)/(Pi*samples_pd(i,:)');
            end
        end

        likelihood_function_value_old=likelihood_function_value;
        likelihood_function_value=sum(log(sum(samples_pd.*repmat(Pi,N,1),1)));
        record=[record,likelihood_function_value];
        beta=abs(likelihood_function_value-likelihood_function_value_old);    
        if beta<1e-8
%             plot(1:length(record),record)
            break
        end
        %% %M��
        Nk=sum(gama,1);
        for j=1:K
            for i=1:N
                
            end
        end
        for j=1:K
            Miu{j,1}=zeros(1,D);
            Sigma2{j,1}=zeros(D,D);
            for i=1:N
                Miu{j,1}=Miu{j,1}+gama(i,j)*samples(i,:)/Nk(j);
            end 
            for i=1:N
                Sigma2{j,1}=Sigma2{j,1}+(gama(i,j)*(samples(i,:)-Miu{j,1})'*(samples(i,:)-Miu{j,1}))/Nk(j);
            end
        end
        Pi=Nk/N;
    end
    
    % exchange components: Sort in acscending order of sigma
    [~, idx] =sort(cell2mat(Sigma2)', 'ascend');
    Pi=Pi(idx);
    Miu = Miu(idx);
    Sigma2 = Sigma2(idx);
    gmm_dist = gmdistribution(vertcat(Miu{:}), cat(3, Sigma2{:}), Pi);
end

function [M]=matrix_ecef2enu(p)
% https://www.cnblogs.com/charlee44/p/15382659.html
    B=p.B*(pi/180);L=p.L*(pi/180);H=p.H;
    Xp=p.Xp;Yp=p.Yp;Zp=p.Zp;
    R=[-sin(L),cos(L), 0, 0;
       -sin(B)*cos(L), -sin(B)*sin(L), cos(B), 0;
       cos(B)*cos(L),cos(B)*sin(L),sin(B),0;
       0,0,0,1];
   T=eye(4);
   T(:,4)=[-Xp,-Yp,-Zp,1];
   M=R*T;
end