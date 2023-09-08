function [intervals, epsilon_achieved] = adjust_sym_uni( sample_data, intervals,epsilon)
%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************

%This function modifies the symmetric unimodal distribution defined by
%intervals so that it is a 1+epsilon right side cdf overbound of the sample
%data
%The current version modifies the symmetric unimodal distribution by
%compressing it keeping the second to last interval lower bound fixed

if nargin<3 
   epsilon = 0;
end
sample_data = sort(sample_data,'ascend');
Nsamples = length(sample_data);

%Place second to last interval lower bound at the location of the last data
%sample
delta_1 = sample_data(end)-intervals(end-1);
intervals = intervals + delta_1 + eps;

%only adjust if necessary
cdf_sym_unimodal = ((Nsamples:-1:1)/Nsamples)'; 
cdf_sample  = compute_cdf( sample_data,intervals(1:(end-1)));
epsilon_lo = max((cdf_sample-cdf_sym_unimodal)./cdf_sample);

if epsilon_lo>epsilon
alpha_lo = 1;
%intervals_lo = intervals;

%for the iteration, we only need the first 1 to end-1 elements
fixed_point = intervals(end-1);
y1 = sample_data(1:end-1)-fixed_point;
y2 = intervals(1:(end-2))-fixed_point;

%upper bound
alpha_hi = min(y1./y2);
%intervals_hi = alpha_hi*intervals +(1-alpha_hi)*fixed_point;
%cdf_sample  = compute_cdf( sample_data,intervals_hi(1:(end-1)));

%epsilon_hi = max((cdf_sample-cdf_sym_unimodal)./cdf_sym_unimodal);
count = 0;
while (epsilon_lo>epsilon)&&(count<10)
    count = count+1;
    alpha_mid = .5*(alpha_lo+alpha_hi);
    intervals_mid = alpha_mid*intervals +(1-alpha_mid)*fixed_point;
    cdf_sample = compute_cdf( sample_data,intervals_mid(1:(end-1)));
    epsilon_mid = max((cdf_sample-cdf_sym_unimodal)./cdf_sample);
    
    if epsilon_mid>epsilon
       alpha_lo = alpha_mid;
       epsilon_lo = epsilon_mid;
    else
        alpha_hi = alpha_mid;
        %epsilon_hi = epsilon_mid;
    end   
end

intervals = alpha_hi*intervals +(1-alpha_hi)*fixed_point;
cdf_sample  = compute_cdf( sample_data,intervals(1:(end-1)));



end
epsilon_achieved = max((cdf_sample-cdf_sym_unimodal)./cdf_sym_unimodal);
