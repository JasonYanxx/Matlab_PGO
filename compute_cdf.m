function [ sampleCdf ] = compute_cdf( sample,x  )

%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************

%This function computes the cdf of the sample distribution defined by the
%sample locations ('sample') at the points x
sample = sort(sample);
Nsample = length(sample);
Nloc = length(x);
y = [x ; sample];
[y_sorted,idx] = sort(y);
[~,idx_sorted] = sort(idx);

sampleCdf = (Nsample - (idx_sorted(1:Nloc)-(1:1:(Nloc))'))/Nsample;

end

