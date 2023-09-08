function [alpha_overbound ] = find_alpha(halfBinEdge, halfSymCdf, NstepsCdf,gama_overbound)

%find_sigma.m finds the sigma overbound of the distribution defined by
%halfBinEdge and halfSymCdf

%   Detailed explanation goes here
if NstepsCdf>0
   [ halfBinEdge,  halfSymCdf ] = reduce_cdf_size( halfBinEdge, halfSymCdf, NstepsCdf );
end

%Lower bound
alpha_min =1;

%Upper bound
alpha_max=2;


sig_up = alpha_max;
sig_lo = alpha_min;
tic
crit = 1;
iteration = 0;
while crit>.01
     iteration = iteration+1;
     sig_mid = .5*(sig_up+sig_lo);
     cdfDiff = evaluate_stable(halfSymCdf, halfBinEdge, gama_overbound,sig_mid);
     if cdfDiff>=0 % watch out!!! here is 'lager than'
         sig_lo = sig_mid;
     else
         sig_up = sig_mid;
     end
     crit = sig_up-sig_lo;
end
alpha_overbound = sig_up;


function [g]=sascdf(x,a,r)
    g=cdf(makedist('Stable','alpha',a,'beta',0,'gam',r,'delta',0),x);
end

function [ minCdfDiff ] = evaluate_stable( sampleCdf, binEdge, gama,alpha)
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
     % tangent
%     y1 = cauchycdf(-x0,gama)-cauchypdf(x0,gama)*(x1-x0);
%     y2 = cauchycdf(-x0,gama)-cauchypdf(x0,gama)*(x2-x0);
     y1 = sascdf(-x1,alpha,gama);
     y2 = sascdf(-x2,alpha,gama);
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


end

