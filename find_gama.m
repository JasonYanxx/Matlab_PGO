function [ gama_overbound ] = find_gama(halfBinEdge, halfSymCdf, NstepsCdf)

%find_sigma.m finds the sigma overbound of the distribution defined by
%halfBinEdge and halfSymCdf

%   Detailed explanation goes here
if NstepsCdf>0
   [ halfBinEdge,  halfSymCdf ] = reduce_cdf_size( halfBinEdge, halfSymCdf, NstepsCdf );
end

%Perform cdf bounding on the right hand side cdf
%find median and mean of symmetric unimodal distribution
[~,index] = max(halfBinEdge(halfSymCdf>0));

%Lower bound
gama_min = -halfBinEdge(index)/cauchyinv(halfSymCdf(index));

%Upper bound
%determine bound for a uniform distribution
p_uniform = .5/max(halfBinEdge);
gama_max = 1/(3.1415*p_uniform);
% title('sample right cdf')

sig_up = gama_max;
sig_lo = gama_min;
tic
crit = 1;
iteration = 0;
while crit>.01
     iteration = iteration+1;
     sig_mid = .5*(sig_up+sig_lo);
     cdfDiff = evaluate_gama(halfSymCdf, halfBinEdge, sig_mid);
     if cdfDiff<=0
         sig_lo = sig_mid;
     else
         sig_up = sig_mid;
     end
     crit = sig_up-sig_lo;
end
gama_overbound = sig_up;


function  [x]=cauchyinv(p)
    x=tan(3.1415*(p-0.5));
end

function [p]=cauchypdf(x,r)
    p=1/(3.1415*r*(1+(x/r)^2));
end

function [g]=cauchycdf(x,r)
    g=(1/3.1415)*atan(x/r)+0.5;
end

function [ minCdfDiff ] = evaluate_gama( sampleCdf, binEdge, gama)
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
     y1 = cauchycdf(-x1,gama);
     y2 = cauchycdf(-x2,gama);
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

