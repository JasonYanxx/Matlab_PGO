%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************
% gene GMM
p1=0.5;
p2=0.5;
% mu1=0.01;
% mu2=-0.01;
% sigma1=0.01^2;
% sigma2=0.01^2;
mu1=0.03;
mu2=-0.01;
sigma1=0.02^2;
sigma2=0.08^2;
gm = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
Xgmm = random(gm, 100000);


% gene GMM - total mean & variance
mut=p1*mu1+p2*mu2;
sigmat=p1*sigma1+(mut-mu1)*(mut-mu1)+p2*sigma2+(mut-mu2)*(mut-mu2);
Xt = normrnd(mut, sqrt(sigmat), 100000, 1);
disp("mut")
disp(mut)
disp("std0")
disp(sqrt(sigmat))
% x = linspace(min(Xt), max(Xt), 100000);
% yt = normpdf(x, mut, sqrt(sigmat));

%%
sample_data=Xgmm;
% load sample_data  %normalized GPS URA clock and ephemeris errors  
close all
figures = 1;

%sample_data = [randn(10000,1)-2;randn(9000,1)+1]; %Bimodal example

Nsamples = length(sample_data);
Nbins = 100;
NstepsCdf = 1000;
epsilon = 0.0025;
   
% Right side overbound
[mean_overbound, sigma_overbound, epsilon_achieved, intervals]=gaussian_overbound(sample_data, epsilon, Nbins, NstepsCdf);
mean_overbound_r = mean_overbound;
sigma_overbound_r = sigma_overbound;
epsilon_r      = epsilon_achieved;

cdf_sym_unimodal = ((Nsamples:-1:1)/Nsamples)';
cdf_sample  = compute_cdf( sample_data,intervals(1:(end-1)));
cdf_gaussian = 1-normcdf(intervals(1:end-1), mean_overbound, sigma_overbound);


%% T1 pp
xbayes = linspace(min(Xgmm), max(Xgmm), 100000);
T1transpp=zeros(1,100000);
for i=1:100000
    y=xbayes(1,i);
    [s1,s2]=cal_omega(y',mu1,sigma1,p1,mu2,sigma2,p2);
    [T1]=cal_T(y,mu1,sigma1,s1,mu2,sigma2,s2);
    T1transpp(1,i)=normpdf(y, s1*mu1+s2*mu2, s1*sqrt(sigma1)+s2*sqrt(sigma2));
end

% culmative prob starting from right
T1transpp_inverse = flip(T1transpp);
cdf_T1transpp = cumtrapz(T1transpp_inverse);
% T1pp_cdf = T1pp_cdf / T1pp_cdf(end);
cdf_T1transpp=cdf_T1transpp*(xbayes(2)-xbayes(1));
cdf_T1transpp=flip(cdf_T1transpp);
% plot(flip(xgmm), cdf_T1transpp, 'LineWidth', 2);

%%
if figures
    plot_cdfs
end

[mean_overbound, sigma_overbound, epsilon_achieved, intervals]=gaussian_overbound(-sample_data, epsilon, Nbins, NstepsCdf);
mean_overbound_l = mean_overbound;
sigma_overbound_l = sigma_overbound;
epsilon_l      = epsilon_achieved;

cdf_sample  = compute_cdf( -sample_data,intervals(1:(end-1)));
cdf_gaussian = 1-normcdf(intervals(1:end-1), mean_overbound, sigma_overbound);
sample_data = -sample_data;

%% T1 pp
Xgmm=-Xgmm;
xbayes = linspace(min(Xgmm), max(Xgmm), 100000);
T1transpp=zeros(1,100000);
for i=1:100000
    y=xbayes(1,i);
    [s1,s2]=cal_omega(y',-mu1,sigma1,p1,-mu2,sigma2,p2);
    [T1]=cal_T(y,-mu1,sigma1,s1,-mu2,sigma2,s2);
    T1transpp(1,i)=normpdf(y, -s1*mu1-s2*mu2, s1*sqrt(sigma1)+s2*sqrt(sigma2));
end

% culmative prob starting from right
T1transpp_inverse = flip(T1transpp);
cdf_T1transpp = cumtrapz(T1transpp_inverse);
% T1pp_cdf = T1pp_cdf / T1pp_cdf(end);
cdf_T1transpp=cdf_T1transpp*(xbayes(2)-xbayes(1));
cdf_T1transpp=flip(cdf_T1transpp);
% plot(flip(xgmm), cdf_T1transpp, 'LineWidth', 2);
%% 

if figures
   plot_cdfs
end

disp('overbound mean    overbound sigma    epsilon')
   [mean_overbound_l sigma_overbound_l epsilon_l;
    mean_overbound_r sigma_overbound_r epsilon_r]
% disp('data mean')
%     mean(sample_data)
% disp('data median')
%     median(sample_data)
% disp('data standard deviation')
%     std(sample_data)

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