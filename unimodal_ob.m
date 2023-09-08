function [ intervals] = unimodal_ob(symLowerBinLimit, symUpperBinLimit,complete_shiftedHistogram)
%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************

Nbins_sym = length(symUpperBinLimit);
symUnimodalSamples = [];
% 根据直方图来重新生成samples
for i=1:Nbins_sym
      symUnimodalSamples =[symUnimodalSamples ...
          symLowerBinLimit(i)+(1:complete_shiftedHistogram(i))*(symUpperBinLimit(i)-symLowerBinLimit(i))/complete_shiftedHistogram(i)];
end
symUnimodalSamples =  [symLowerBinLimit(1) symUnimodalSamples]';

%transform into a unimodal distribution
%first find midpoints between samples
midpoints = .5*(symUnimodalSamples(1:end-1)+symUnimodalSamples(2:end));

%add one point at the beginning and the end
intervals = [(symUnimodalSamples(1) - (midpoints(1) - symUnimodalSamples(1))); ...
              midpoints; ...
             (symUnimodalSamples(end) + (midpoints(1) - symUnimodalSamples(1)))];        

%shift intervals to so that we still bound the last sample to the right
intervals = intervals + (midpoints(1) - symUnimodalSamples(1));


end

