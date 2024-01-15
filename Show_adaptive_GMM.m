% 创建一个图形窗口
clear all
close all
figure;

% 设置动画的时间步长和总时间
st= 30;
dt = 5;
T = 45;

% 加载数据
load('Data/urban_dd_0816/mergeurbandd.mat');
filter_err=(mergedurbandd.doubledifferenced_pseudorange_error>=-15 & mergedurbandd.doubledifferenced_pseudorange_error<=15); 
filter_ele=(mergedurbandd.U2I_Elevation>=st & mergedurbandd.U2I_Elevation<=st+dt);
Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele & filter_err);
Nsamples=length(Xdata);
lim=max(-min(Xdata),max(Xdata));
x = linspace(-lim, lim, Nsamples);
% 绘制初始图形
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

% 循环更新数据和绘图
for t = dt:dt:T
    % 更新数据
    alpha=1+t;
    
    % 清除之前的图形
    clf;
    
    % 绘制新的图形
    filter_err=(mergedurbandd.doubledifferenced_pseudorange_error>=-15 & mergedurbandd.doubledifferenced_pseudorange_error<=15); 
    filter_ele=(mergedurbandd.U2I_Elevation>=st & mergedurbandd.U2I_Elevation<=st+dt+t);
    Xdata=mergedurbandd.doubledifferenced_pseudorange_error(filter_ele & filter_err);
    Nsamples=length(Xdata);
    lim=max(-min(Xdata),max(Xdata));
    x = linspace(-lim, lim, Nsamples);
    % 绘制初始图形
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
    
    % 将当前帧保存为GIF文件的一部分
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if t == dt
        imwrite(imind, cm, 'gmmfit.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
    else
        imwrite(imind, cm, 'gmmfit.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
    end
    
    % 暂停一段时间以使动画看起来更平滑
    pause(0.3);
end