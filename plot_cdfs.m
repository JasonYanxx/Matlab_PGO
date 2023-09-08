%*************************************************************************
%*     Copyright c 2017 The board of trustees of the Leland Stanford     *
%*                      Junior University. All rights reserved.          *
%*     This script file may be distributed and used freely, provided     *
%*     this copyright notice is always kept with it.                     *
%*                                                                       *
%*     Questions and comments should be directed to Juan Blanch at:      *
%*     blanch@stanford.edu                                               *
%*************************************************************************

%This script plots the cdfs and pdfs of the three following three
%distributions: sample distribution, symmetric unimodal upper bound,
%gaussian overbound

figure
plot(intervals(1:end-1), cdf_sym_unimodal - cdf_sample)
xlabel('quantile')
ylabel('cdf symmetric unimodal - sample cdf')

%compute cdf of gaussian overbound
cdf_gaussian = 1-normcdf(intervals(1:end-1), mean_overbound, sigma_overbound);

figure
semilogy(intervals(1:end-1), cdf_sample,'LineWidth',2)
hold on
semilogy(intervals(1:end-1), cdf_sym_unimodal,'r','LineWidth',2)
hold on
semilogy(intervals(1:end-1), cdf_gaussian,'g','LineWidth',2)
%% T1transpp
hold on
semilogy(xbayes, cdf_T1transpp,'b','LineWidth',2)
yline(0.05)
yline(0.01)
yline(0.001)
%% 
set(gca,'FontSize',12);
xlabel('quantile','FontSize',12)
ylabel('cdf','FontSize',12)
A = legend('sample distribution','sym. uni. bounding dist.','gaussian bounding dist.','T1transpp');
set(A,'FontSize',12)

x = min(intervals):.01:max(intervals);
y = normpdf(x,mean_overbound,sigma_overbound);

%y_su = compute_cdf(x,intervals, ones(length(intervals),1));
%[y_su x_su] = histcounts(intervals,'normalization','pdf');
N = length(intervals);
y_su = zeros(N-1,1);
x_su = intervals(2:N);

for i=1:length(intervals)-1
    
    dx = intervals(i+1)-intervals(i);
    y_su(i) = 1/((N-1)*dx);
end

figure; 
histogram(sample_data,'normalization','pdf')
xlim([min(intervals)  max(intervals)]);
xlabel('normalized GPS clock and ephemeris error','FontSize', 14)
ylabel('pdf','FontSize',14)
hold on
plot(x_su,y_su,'r','LineWidth',2)
hold on
plot(x,y,'g','LineWidth',2)
%% T1transpp
hold on
plot(xbayes,T1transpp,'b','LineWidth',2)
%%
A = legend('sample distribution','sym. uni. bounding dist.','gaussian bounding dist.','T1transpp');
set(A,'FontSize',12)