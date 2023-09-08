function [ sigma_overbound ] = find_sigma(halfBinEdge, halfSymCdf, NstepsCdf)

%find_sigma.m finds the sigma overbound of the distribution defined by
%halfBinEdge and halfSymCdf

%   Detailed explanation goes here
if NstepsCdf>0
   [ halfBinEdge,  halfSymCdf ] = reduce_cdf_size( halfBinEdge, halfSymCdf, NstepsCdf );
end

%Perform cdf bounding on the right hand side cdf
%find median and mean of symmetric unimodal distribution
[~,i] = max(halfBinEdge(halfSymCdf>0));

%Lower bound
sigma_min = -halfBinEdge(i)/norminv(halfSymCdf(i));

%Upper bound
%determine bound for a uniform distribution
p_uniform = .5/max(halfBinEdge);
sigma_max = 1/(sqrt(2)*p_uniform);
% title('sample right cdf')

sig_up = sigma_max;
sig_lo = sigma_min;
tic
crit = 1;
iteration = 0;
while crit>.01
     iteration = iteration+1;
     sig_mid = .5*(sig_up+sig_lo);
     cdfDiff = evaluate_sigma(halfSymCdf, halfBinEdge, sig_mid);
     if cdfDiff<=0
         sig_lo = sig_mid;
     else
         sig_up = sig_mid;
     end
     crit = sig_up-sig_lo;
end
sigma_overbound = sig_up;



end

