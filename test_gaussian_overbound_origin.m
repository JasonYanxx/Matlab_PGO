%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************



load sample_data  %normalized GPS URA clock and ephemeris errors  
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
