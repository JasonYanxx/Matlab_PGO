clear all
close all
figure;

% set step size (dt) and total time (T) for animation
st= 30;
dt = 5;
T = 45;

load('Data/urban_dd_0816/mergeurbandd.mat');
filter_err=(mergedurbandd.doubledifferenced_pseudorange_error>=-15 & mergedurbandd.doubledifferenced_pseudorange_error<=15); 
filter_ele=(mergedurbandd.U2I_Elevation>=st & mergedurbandd.U2I_Elevation<=st+dt);
Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele & filter_err);
Nsamples=length(Xdata);
lim=max(-min(Xdata),max(Xdata));
x = linspace(-lim, lim, Nsamples);
% plot initial figure
histogram(Xdata,'normalization','pdf');
hold on
% EM fitting
options = statset('TolFun', 1e-6, 'MaxIter', 10000);
gmm_dist = fitgmdist(Xdata, 4, 'Options', options);
fit_pdf=pdf(gmm_dist,x');
plot(x,fit_pdf','LineWidth',4)
axis([-15 15 0 0.25]);
ylabel('PDF','FontSize',18);
set(gca, 'FontSize', 18,'FontName', 'Times New Roman');
drawnow;
pause(0.3);

% update
for t = dt:dt:T
    % update data
    alpha=1+t;
    
    % clear history plots
    clf;
    
    % plot new figures
    filter_err=(mergedurbandd.doubledifferenced_pseudorange_error>=-15 & mergedurbandd.doubledifferenced_pseudorange_error<=15); 
    filter_ele=(mergedurbandd.U2I_Elevation>=st & mergedurbandd.U2I_Elevation<=st+dt+t);
    Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele & filter_err);
    Nsamples=length(Xdata);
    lim=max(-min(Xdata),max(Xdata));
    x = linspace(-lim, lim, Nsamples);
    % plot initial figure
    histogram(Xdata,'normalization','pdf');
    hold on
    % EM fitting
    options = statset('TolFun', 1e-6, 'MaxIter', 10000);
    gmm_dist = fitgmdist(Xdata, 4, 'Options', options);
    fit_pdf=pdf(gmm_dist,x');
    plot(x,fit_pdf','LineWidth',4)
    axis([-15 15 0 0.25]);
    ylabel('PDF','FontSize',18);
    set(gca, 'FontSize', 18,'FontName', 'Times New Roman');
    drawnow;
    
    % save current snapshot into GIF
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if t == dt
        imwrite(imind, cm, 'gmmfit.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
    else
        imwrite(imind, cm, 'gmmfit.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
    end
    
    % make the animation more smooth
    pause(0.3);
end