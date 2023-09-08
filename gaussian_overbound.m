function [mean_overbound, sigma_overbound, epsilon_achieved, intervals]= gaussian_overbound(sample_data,epsilon, Nbins, NstepsCdf)

%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************
%This function implements an evolution of the algorithm described in "A
%Method to Determine Strict Gaussian Bounding Distributions of a Sample Distribution"
%ION GNSS2017

%INPUTS
%sample_data is the set of samples describing the sample distribution
%figures is a flag to plot the figures illustrating the process
%epsilon is the maximum allowable excess mass
%Nbins is used to form the histogram used to force the unimodalization.  It
%is initialized at 200. 
%NstepsCdf is the number of steps used in the discretization of the cdf. It
%is initialized at 500.

%OUTPUTS
%mean_overbound and sigma_overbound are the parameters of the right side
%gaussian overbound
%epsilon_achieved is the achieved excess mass.  It is output for
%verification purposes
%intervals defined the intermediate symmetric unimodal distribution at
%intervals(0)=1 and intervals(end)=0. The cdf decreases by 1/Nsamples over
%each interval [interval(i) interval(i+1)]


if nargin<2
    epsilon = 0.0025;
end
if nargin<3
    Nbins = 200;
end
if nargin<4
    NstepsCdf = 500;
end

Nsamples = length(sample_data);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bin the sample data %%%%%%%%%%%%%%%%%%%%%%%%%%

[lowerBinLimit, upperBinLimit, sampleHistogram] = bin_sample_dist(sample_data, ones(length(sample_data),1), Nbins);


%%%%%%%% Determine symmetric and quasi-unimodal right side upper bound %%%%

[symLowerBinLimit,symUpperBinLimit,complete_shiftedHistogram ] = sym_unimodal_shape(lowerBinLimit, upperBinLimit, sampleHistogram);


%%%%%%%%%%%%%%%%% Distribute samples evenly within each bin %%%%%%%%%%%%%%%

[ intervals] = unimodal_ob(symLowerBinLimit, symUpperBinLimit,complete_shiftedHistogram);

%At the end of this step we have obtained the shape of the overbounding
%distribution

%%%%%%%%%%%%%%%%%%%% Adjust the sym. uni. distribution   %%%%%%%%%%%%%%%%%%

[ intervals, epsilon_achieved] = adjust_sym_uni( sample_data, intervals, epsilon);


%%%%%%%%%%%   Perform gaussian bounding on the right hand side cdf  %%%%%%%

i_median = ceil((Nsamples+1)/2);
mean_overbound = intervals(i_median);

cdf_sym_unimodal = ((Nsamples:-1:1)/Nsamples)';

halfBinEdge = intervals(i_median:end)-intervals(i_median);
halfSymCdf  = [cdf_sym_unimodal(i_median:end); 0];
sigma_overbound = find_sigma(halfBinEdge, halfSymCdf, NstepsCdf);

