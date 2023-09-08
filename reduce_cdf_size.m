function [ NewHalfBinEdge,  NewHalfSymCdf ] = reduce_cdf_size( halfBinEdge, halfSymCdf, NewNbins )
%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************

%ReduceCdfSize reduces the size of the right hand side cdf to the specified number of bins
%The function selects a subset of edges evenly distributed distributed.
Nbins = length(halfBinEdge);

idx=1;
BinWidth = (halfBinEdge(end)-halfBinEdge(1))/NewNbins;
k=2;
indices = NaN(NewNbins,1);
indices(1) = 1;
while (halfBinEdge(end)-halfBinEdge(idx))>BinWidth
    iplus = find(halfBinEdge>=BinWidth*(k-1));
    [~,idx] = min(halfBinEdge(iplus));  
    idx = iplus(idx);
    indices(k) = idx;
    k = k+1;
end

if idx<Nbins
indices(k) = Nbins;
indices = indices(1:k);
else
   indices = indices(1:(k-1)); 
end

NewHalfBinEdge = halfBinEdge(indices);
NewHalfSymCdf = halfSymCdf(indices);