function [lowerBinLimit, upperBinLimit, sampleHistogram] = bin_sample_dist(x, y, Nbins)
%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************
%This function bins the data in Nbins number of bins


%size of bins
lowerLimit = min(x)-1e-6;
upperLimit = max(x)+1e-6;
binSize = (upperLimit-lowerLimit)/Nbins;

lowerBinLimit = lowerLimit + (0:Nbins-1)*binSize;
upperBinLimit = [lowerBinLimit(2:end) upperLimit];

sampleHistogram = zeros(Nbins,1);

%bin the data 
for i=1:Nbins
    %idx = find(x>lowerBinLimit(i)&sample_data<=upperBinLimit(i));
    sampleHistogram(i) = sum(y((x>lowerBinLimit(i)&x<=upperBinLimit(i))));
end

