clear all;
close all;

rng(3);
%Ad = [0.8 0 0;0 1.2 0;0 0 0.7];
Ad = [1.2 -0.2 -0.1;0.1 1.4 0.2;0.3 0.1 0.7];
%Ad = rand(3,3);
%Bd = [1;1;1];
Bd = rand(3,1); 
Cd = [1 0 0];
T = 500;
k = rank([Ad Ad*Bd]);

%Knom = lqrd(Ad,Bd,2*eye(3),2,1);
clvalues = [0.1+0.1i;0.1-0.1i;-0.1];
Knom = place(Ad,Bd,clvalues);

%margin(d2c(ss(Ad,Bd,Knom,0,1)))

%[KROB,CL,GAM,INFO] = hinfsyn(ss(Ad,[eye(3) Bd],[sqrt(2)*eye(3);eye(3)],[[eye(3) sqrt(0.02/3)*ones(3,1)];[zeros(3,3) zeros(3,1)]],1),3,1);
%Krob = -1*INFO.Ku;

clvalues = [0.6+0.5i;0.6-0.5i;-0.5];
Krob = place(Ad,Bd,clvalues);
kr1 = 1/(Cd*inv(eye(3)-(Ad-Bd*Knom))*Bd);
kr2 = 1/(Cd*inv(eye(3)-(Ad-Bd*Krob))*Bd);

%[KROB,CL,GAM,INFO] = hinfsyn(ss(Ad,[eye(3) Bd],[sqrt(2)*eye(3);eye(3)],[[eye(3) sqrt(0.02/3)*ones(3,1)];[zeros(3,3) zeros(3,1)]],1),3,1);
%Krob = -1*INFO.Ku;

%figure
%pzmap(ss(Ad-Bd*Knom,Bd*kr1,Cd,0));
%figure
%pzmap(ss(Ad-Bd*Krob,Bd*kr2,Cd,0));
[n1,d1] = ss2tf(Ad-Bd*Knom,Bd*kr1,Cd,0,1);
[n2,d2] = ss2tf(Ad-Bd*Krob,Bd*kr2,Cd,0,1);
theta = 0:0.01:2*pi;
figure
testsys = tf(d1,d2,1);

% Plotting parameters
lw = 2;
lwt = 1.5;
ms = 8;
fs = 12;
% ROOT LOCUS vs ZETA
figure(1)
clf
wn=1;
for alpha=[0:0.01:1];
    r=roots(alpha*d1+(1-alpha)*d2);
    if(alpha==0)
      plot(real(r),imag(r),'r.','MarkerSize',2*ms)
    elseif (alpha==1)
      plot(real(r),imag(r),'g.','MarkerSize',2*ms)
    else
      plot(real(r),imag(r),'k.','MarkerSize',ms)  
    end
    if alpha==0 
      hold on
    end
end

hold off
axis([-1 1 -1 1])
zgrid
axis('equal')
xlabel('Real Axis', 'fontsize', fs)
ylabel('Imag Axis', 'fontsize', fs)
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 4.95, 4.95], 'PaperUnits', 'Inches', 'PaperSize', [4.95, 4.95])
saveas(gcf,'rootlocusgeneral.eps','epsc')

sys1 = ss(d2c(ss(Ad,Bd,Knom,0,1)));
sys2 = ss(d2c(ss(Ad,Bd,Krob,0,1)));
sys3 = ss(d2c(ss(Ad,Bd,0.5*Knom+0.5*Krob,0,1)));

figure
nyquist(sys1,sys2,sys3);
xlim([-10 5]);
ylim([-10 10]);
legend('Sys1','Sys2','CVX');
