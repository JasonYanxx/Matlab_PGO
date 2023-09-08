% ����һ��ͼ�δ���
clear all
close all
figure;

% ���ö�����ʱ�䲽������ʱ��
dt = 0.03;
T = 0.97;

% ��ʼ������
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

% ���Ƴ�ʼͼ��
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

% ѭ���������ݺͻ�ͼ
for t = dt:dt:T
    % ��������
    alpha=1+t;
    
    % ���֮ǰ��ͼ��
    clf;
    
    % �����µ�ͼ��
    plot(x,y_gm,'LineWidth',2);
    hold on
    plot(x,y_norm,'LineWidth',2);
    y_stable=cdf(makedist('Stable','alpha',alpha,'beta',0,'gam',gama,'delta',0),x);
    plot(x,y_stable,'LineWidth',2);
    plot(x,sign(y_stable-y_gm'),'LineWidth',2);
    A = legend('sample dist.','Gaussian','stable bound','sign');
    set(A,'FontSize',12,'Location','southeast');
    % ���������᷶Χ
    axis([-1.5 1.5 -1.1 1.1]);
    title(sprintf('alpha = %.2f', alpha));
    drawnow;
    
    % ����ǰ֡����ΪGIF�ļ���һ����
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if t == dt
        imwrite(imind, cm, 'animation.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
    else
        imwrite(imind, cm, 'animation.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
    end
    
    % ��ͣһ��ʱ����ʹ������������ƽ��
    pause(0.3);
end