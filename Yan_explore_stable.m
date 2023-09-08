% 创建一个图形窗口
clear all
close all
figure;

% 设置动画的时间步长和总时间
dt = 0.03;
T = 0.97;

% 初始化数据
p1=0.7;
p2=1-p1;
mu1=0;
mu2=0;
sigma1=0.1342^2;
sigma2=0.4399^2;

gm = gmdistribution([mu1; mu2], cat(3, sigma1, sigma2), [p1 p2]);
Xdata = random(gm, 10001);
Nsamples = length(Xdata);
lim=max(-min(Xdata),max(Xdata));
x = linspace(-lim, lim, Nsamples);


alpha=1;
gama= 0.1475;

% 绘制初始图形
y_gm=cdf(gm, x');
y_norm=normcdf(x,0,0.2873);
plot(x,y_gm,'LineWidth',2);
hold on
plot(x,y_norm,'LineWidth',2);
y_stable=cdf(makedist('Stable','alpha',alpha,'beta',0,'gam',gama,'delta',0),x);
plot(x,y_stable,'LineWidth',2);
plot(x,sign(y_stable-y_gm'),'LineWidth',2);
A = legend('sample dist.','Gaussian','stable bound','sign');
set(A,'FontSize',12,'Location','southeast');
axis([-1.5 1.5 -1.1 1.1]);
title(sprintf('alpha = %.2f', alpha));
drawnow;
pause(0.3);

% 循环更新数据和绘图
for t = dt:dt:T
    % 更新数据
    alpha=1+t;
    
    % 清除之前的图形
    clf;
    
    % 绘制新的图形
    plot(x,y_gm,'LineWidth',2);
    hold on
    plot(x,y_norm,'LineWidth',2);
    y_stable=cdf(makedist('Stable','alpha',alpha,'beta',0,'gam',gama,'delta',0),x);
    plot(x,y_stable,'LineWidth',2);
    plot(x,sign(y_stable-y_gm'),'LineWidth',2);
    A = legend('sample dist.','Gaussian','stable bound','sign');
    set(A,'FontSize',12,'Location','southeast');
    % 设置坐标轴范围
    axis([-1.5 1.5 -1.1 1.1]);
    title(sprintf('alpha = %.2f', alpha));
    drawnow;
    
    % 将当前帧保存为GIF文件的一部分
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if t == dt
        imwrite(imind, cm, 'animation.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
    else
        imwrite(imind, cm, 'animation.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
    end
    
    % 暂停一段时间以使动画看起来更平滑
    pause(0.3);
end