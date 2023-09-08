function [symLowerBinLimit,symUpperBinLimit,symUniHistogram ]= sym_unimodal_shape(lowerBinLimit, upperBinLimit, sampleHistogram)
%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************

Nbins = length(sampleHistogram);
Nsamples = sum(sampleHistogram);
binSize = upperBinLimit(1)-lowerBinLimit(1);
%shift the pdf of the bins to force unimodality on the right side of the
%cdf
shiftedHistogram = sampleHistogram;
cdf = 0;
i=Nbins;
added_mass = 0;

while  cdf<.5*Nsamples    
    newvalue = max(sampleHistogram(i-1),shiftedHistogram(i));
    %cdf=cdf+newvalue;
    cdf = sum(shiftedHistogram(i:end))+newvalue;
    if cdf<.5*Nsamples
       i=i-1; % 选中左侧bin
       shiftedHistogram(i)=newvalue; % 左侧的bin value 一定要比右侧大
       added_mass = newvalue-sampleHistogram(i)+ added_mass;
    end   
end

%determine value of center bin: 
centerValue = Nsamples - 2*sum(shiftedHistogram(i:end))-1; % 为什么要减1

%complete right side with mirror image
symUniHistogram = [shiftedHistogram(end:-1:i);centerValue;shiftedHistogram(i:end)];

%determine minimum width of center bin to guarantee unimodality
%shiftedHistogram(i)= shiftedHistogram(i)-1;
minWidth = binSize*centerValue/(shiftedHistogram(i));
shift = upperBinLimit(Nbins)-lowerBinLimit(i)+minWidth;

%complete left side bins
symLowerBinLimit = [lowerBinLimit(i:Nbins)-shift lowerBinLimit(i)-minWidth lowerBinLimit(i:Nbins)];
symUpperBinLimit = [symLowerBinLimit(2:end) upperBinLimit(end)];

