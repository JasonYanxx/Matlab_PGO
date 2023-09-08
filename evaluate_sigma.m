function [ minCdfDiff ] = evaluate_sigma( sampleCdf, binEdge, sigma)
%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************


% This function calculates an upper bound of the max difference
%between the sample cdf and N(0,sigma)cdf.  It will be an overbound if the
%difference is positive
%sampleCdf: N vector
%binEdge: N vector
%sigma: scalar
Nbins = length(binEdge);

cdfDifference = zeros(Nbins-1,1);

%for first bin, check that the slope of the gaussian is larger than the
%slope of the uniform distribution 

for i=1:Nbins-1

     x1 = binEdge(i);
     x2 = binEdge(i+1);
     x0=.5*(x1+x2);
     if i==1
         x0 = x1;
     end
     y1 = normcdf(-x0/sigma)-normpdf(x0/sigma)/sigma*(x1-x0);
     y2 = normcdf(-x0/sigma)-normpdf(x0/sigma)/sigma*(x2-x0);
     if i==1
     cdfDifference(i) =y2 - sampleCdf(i+1);
     else
        cdfDifference(i)=min((y1-sampleCdf(i)), (y2-sampleCdf(i+1)));       
     end
end

% figure
% plot(binEdge,cdfDifference)
% title('Gaussian Overbound minus symmetrized sample distribution')

minCdfDiff = min(cdfDifference);




end

