clear all;
close all;

rng(5);
%rng(9);
%Ad = [0.8 0 0;0 1.2 0;0 0 0.7];
Ad = rand(3,3);
%Ad = [1.01 -0.4 -0.3;0.1 1.1 -0.2;0.3 0.1 0.7];
Bd = rand(3,1);
Cd = [1 0 0];
T = 500;
k = rank([Ad Ad*Bd]);

%perturb = rand(3,3);
%perturb1 = 0.1*perturb/norm(perturb);
%Aest1 = Ad+perturb1;
perturb2 = rand(3,3);
Aest2 = Ad+1.8*perturb2/norm(perturb2);
smooth = ones(1,T);
smooth(1:T-100) = linspace(0,1,T-100);
Topt = 10;

xt = [0;0;0];
d = 3;
n = 1;
m = 3;
%state = zeros(d+d*d+d*n,1);
%state(d+1:d+d*d) = Aest2(:);
%state(d+d*d+1:d+d*d+d*n) = Bd(:);
%staten = zeros(d+d*d+d*n,1);
%Pstate = 0.5*eye(d+d*d+d*n,d+d*d+d*n);
%Pstaten = zeros(d+d*d+d*n,d+d*d+d*n);
%noisecov = zeros(d+d*d+d*n,d+d*d+d*n);
%noisecov(1:d,1:d) = 0.05*eye(d);
%noisecov(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = 0.0001*eye(d*d+d*n);
%R = 0.05*eye(m);
%F = zeros(d+d*d+d*n,d+d*d+d*n);
%H = zeros(d,d+d*d+d*n);

norms = zeros(T,1);
inputnorm1 = zeros(T,1);
specopt = zeros(T,1);
merroropt = zeros(T,1);
states = zeros(T,1);
ref = zeros(T,1);
yerrs = zeros(T,1);
% ref(100:end) = 20;
% ref(350:end) = 10;
ref = 20*sin(0.01*[0:T]);

wt = zeros(3,T);
et = zeros(T,1);

for t=1:T
  wt(:,t) = normrnd(0,0.2,[3,1]);
  %wt(:,t) = zeros(3,1);
  %et(t) = normrnd(0,0.1);
  et(t) = 0;
end
for t=1:T
  %Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad;  
  Adtrue = Ad;
  Apred = (1-smooth(t))*Aest2+smooth(t)*Ad;
  Bpred = Bd;
  % Unpacking the state
  %Apred = reshape(state(d+1:d+d*d),d,d);
  %Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
  %Apred = (1-smooth(t))*Aest2+smooth(t)*Ad
  %Bpred = Bd;
  % Designing the controller
   if rem(t,Topt)==1
     if t~=1
       Kold = Knom;
       krold = kr;
     end
     %Knom = lqrd(Apred,Bpred,2*eye(3),0.2*eye(n),1);
     %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(3) Bpred],[2*eye(3);zeros(n,3);eye(3)],[[zeros(3,3) zeros(3,n)];[zeros(n,3) zeros(n,n)];[zeros(3,3) zeros(3,n)]],1),3,n);
     %Knom = dlqr(Apred,Bpred,20*eye(3),0.2*eye(n)); 
     %Knom = -1*INFO.Ku;
     clvalues = [0.1+0.1i;0.1-0.1i;-0.1];
     Knom = place(Apred,Bpred,clvalues);
     kr = 1/(Cd*inv(eye(3)-(Apred-Bpred*Knom))*Bpred);
     if t==1
       Kold = Knom;
       krold = kr;
     end
     weight = 1;
   end
  
  % Finding the input
  xtold = weight*(-Kold*xt+krold*ref(t)+Knom*xt-kr*ref(t));
  u = xtold-Knom*xt+kr*ref(t);
  inputnorm1(t) = norm(u);
  weight = 0.9*weight;
  %u = normrnd(0,1);
  % Finding the F-jacobian
  %F(1:d,1:d) = Apred;
  %F(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = eye(d*d+d*n);
  %for iu=1:d
  %  F(1:d,d+(iu-1)*d+1:d+iu*d) = state(iu)*eye(d);  
  %end
  %for inu=1:n
  %  F(1:d,d+d*d+(inu-1)*d+1:d+d*d+inu*d) = (u(inu))*eye(d);  
  %end

  % Maintaining our prediction of the system
  %staten = state;
  %staten(1:d) = Apred*state(1:d)+Bpred*u;  
  %Pstaten = F*Pstate*F';
  %Pstaten = Pstaten+noisecov;
  %xpred = staten(1:d);
  
  xt = Adtrue*xt+Bd*u+wt(:,t);
  yt = Cd*xt+et(t);
  %yerror = xt-xpred;
  %yerrs(t) = norm(yerror);
  states(t,:) = yt;
  norms(t) = norm(xt);
  specopt(t) = max(abs(eig(Adtrue-Bd*Knom)));
  merroropt(t) = norm(Apred-Adtrue);
  
  %H(1:m,1:d) = eye(m);
  %Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
  %state = staten+Kt*yerror;
  %Pstate = Pstaten-Kt*H*Pstaten;
end
%figure
%plot(yerrs);

figure
plot(states(:,1));
%%
xt = [0;0;0];
d = 3;
n = 1;
m = 3;
%state = zeros(d+d*d+d*n,1);
%state(d+1:d+d*d) = Aest2(:);
%state(d+d*d+1:d+d*d+d*n) = Bd(:);
%staten = zeros(d+d*d+d*n,1);
%Pstate = 0.5*eye(d+d*d+d*n,d+d*d+d*n);
%Pstaten = zeros(d+d*d+d*n,d+d*d+d*n);
%noisecov = zeros(d+d*d+d*n,d+d*d+d*n);
%noisecov(1:d,1:d) = 0.05*eye(d);
%noisecov(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = 0.0001*eye(d*d+d*n);
%R = 0.05*eye(m);
%F = zeros(d+d*d+d*n,d+d*d+d*n);
%H = zeros(d,d+d*d+d*n);

norms2 = zeros(T,1);
inputnorm2 = zeros(T,1);
specrob = zeros(T,1);
merrorrob = zeros(T,1);
states2 = zeros(T,1);
ref = zeros(T,1);
yerrs2 = zeros(T,1);
% ref(100:end) = 20;
% ref(350:end) = 10;
ref = 20*sin(0.01*[0:T]);
Trob = 10;

for t=1:T
  %Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad;  
  Adtrue = Ad;
  Apred = (1-smooth(t))*Aest2+smooth(t)*Ad;
  Bpred = Bd;
  % Unpacking the state
  %Apred = reshape(state(d+1:d+d*d),d,d);
  %Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
  %Apred = (1-smooth(t))*Aest2+smooth(t)*Ad
  %Bpred = Bd;
  % Designing the controller
  if rem(t,Trob)==1
     if t~=1
       Kold = Krob;
       krold = kr;
     end
     %Knom = lqrd(Apred,Bpred,2*eye(3),0.2*eye(n),1);
     %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(3) Bpred],[2*eye(3);zeros(n,3);eye(3)],[[zeros(3,3) zeros(3,n)];[zeros(n,3) zeros(n,n)];[zeros(3,3) zeros(3,n)]],1),3,n);
     %Knom = dlqr(Apred,Bpred,20*eye(3),0.2*eye(n)); 
     %Knom = -1*INFO.Ku;
     clvalues = [0.5+0.3i;0.5-0.3i;-0.5];
     Krob = place(Apred,Bpred,clvalues);
     kr = 1/(Cd*inv(eye(3)-(Apred-Bpred*Krob))*Bpred);
     if t==1
       Kold = Krob;
       krold = kr;
     end
     weight = 1;
   end
  
  % Finding the input
  xtold = weight*(-Kold*xt+krold*ref(t)+Krob*xt-kr*ref(t));
  u = xtold-Krob*xt+kr*ref(t);
  inputnorm1(t) = norm(u);
  weight = 0.9*weight;
  %u = normrnd(0,1);
  % Finding the F-jacobian
  %F(1:d,1:d) = Apred;
  %F(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = eye(d*d+d*n);
  %for iu=1:d
  %  F(1:d,d+(iu-1)*d+1:d+iu*d) = state(iu)*eye(d);  
  %end
  %for inu=1:n
  %  F(1:d,d+d*d+(inu-1)*d+1:d+d*d+inu*d) = (u(inu))*eye(d);  
  %end

  % Maintaining our prediction of the system
  %staten = state;
  %staten(1:d) = Apred*state(1:d)+Bpred*u;  
  %Pstaten = F*Pstate*F';
  %Pstaten = Pstaten+noisecov;
  %xpred = staten(1:d);
  
  xt = Adtrue*xt+Bd*u+wt(:,t);
  yt = Cd*xt+et(t);
  %yerror = xt-xpred;
  %yerrs(t) = norm(yerror);
  states2(t,:) = yt;
  norms2(t) = norm(xt);
  specrob(t) = max(abs(eig(Adtrue-Bd*Krob)));
  merrorrob(t) = norm(Apred-Adtrue);
  
  %H(1:m,1:d) = eye(m);
  %Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
  %state = staten+Kt*yerror;
  %Pstate = Pstaten-Kt*H*Pstaten;
end
 %%
 norms3 = zeros(T,1);
 inputnorm3 = zeros(T,1);

 yerrs3 = zeros(T,1);
 states3 = zeros(T,1);
 speccvx = zeros(T,1);
 merrorcvx = zeros(T,1);

 Trob = 10;
 Topt = 10;
 rhodes = 0.3;
 Talpha = 5;
 alpha = 1;
 alphavals = [alpha];
 iter = 1;
 
 xt = [0;0;0];
 d = 3;
 n = 1;
 m = 3;
 state = zeros(d+d*d+d*n,1);
 %state(1:d) = normrnd(0,1,[d,1]);
 state(d+1:d+d*d) = Aest2(:);
 state(d+d*d+1:d+d*d+d*n) = Bd(:);
 staten = zeros(d+d*d+d*n,1);
 Pstate = 0.5*eye(d+d*d+d*n,d+d*d+d*n);
 Pstaten = zeros(d+d*d+d*n,d+d*d+d*n);
 noisecov = zeros(d+d*d+d*n,d+d*d+d*n);
 noisecov(1:d,1:d) = 0.05*eye(d);
 noisecov(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = 0.0001*eye(d*d+d*n);
 R = 0.05*eye(m);
 F = zeros(d+d*d+d*n,d+d*d+d*n);
 H = zeros(d,d+d*d+d*n);
 delta = 5;
 for t=1:T
     %Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad; 
     % Unpacking the state
     %Apred = reshape(state(d+1:d+d*d),d,d);
     %Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
     Adtrue = Ad;
     Apred = (1-smooth(t))*Aest2+smooth(t)*Ad;
     Bpred = Bd;
     if rem(t,Topt)==1
       %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(3) Bpred],[2*eye(3);zeros(n,3);eye(3)],[[zeros(3,3) zeros(3,n)];[zeros(n,3) zeros(n,n)];[zeros(3,3) zeros(3,n)]],1),3,n);
       %Knom = dlqr(Apred,Bpred,20*eye(3),0.2*eye(n)); 
       %Knom = -1*INFO.Ku;
       clvalues = [0.1+0.1i;0.1-0.1i;-0.1];
       Knom = place(Apred,Bpred,clvalues);
     end
     
     if rem(t,Trob)==1
       %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(3) Bpred],[sqrt(0)*eye(3);zeros(n,3);eye(3)],[[zeros(3,3) zeros(3,n)];[zeros(n,3) sqrt(0.2)*ones(n,n)];[zeros(3,3) zeros(3,n)]],1),3,n);
       %Krob = -1*INFO.Ku;
       clvalues = [0.5+0.3i;0.5-0.3i;-0.5];
       Krob = place(Apred,Bpred,clvalues);
     end
     
     if rem(t,Talpha)==1
       if t~=1
         Kold = Kcvx;
         krold = kr;
       end  
       Kcvx = alpha*Krob+(1-alpha)*Knom;
       M = Apred-Bpred*Kcvx;
       [V,D,W] = eig(M);
       [~,index] = max(abs(diag(D)));
       u = V(:,index);
       v = W(:,index);
       r = abs(D(index,index));
       S = 1/abs(v'*u);
       Sn = min(norm(S),0.1);
       %Sn = 0.001*norm(S)
       %Sn = 0;
       s = real(conj(D(index,index)/r)*(v'*Bpred*(Knom-Krob)*u)/(v'*u));
       %s = sign(s)*max(abs(s),1);
       delta = norm((reshape(diag(abs(Pstate(d+1:d+d*d,d+1:d+d*d))),d,d)));
       %delta = norm(Ad-Apred)
       % if(t~=1)
       %  delta = yerrs3(t-1);
       %end
       %alpha = 0.98*alpha + 0.02*(rhodes-r-Sn*delta)/s;
       rhodiff = (rhodes-r-Sn*delta);
       alpha = alpha+0.3*rhodiff*s;
       if alpha>1
         alpha = 1;
       elseif alpha<0
         alpha = 0;
       end
       alphavals = [alphavals alpha];
       Kcvx = alpha*Krob+(1-alpha)*Knom;
       kr = pinv(Cd*inv(eye(3)-(Apred-Bpred*Kcvx))*Bpred);
       %Kcvx
       %Krob
       if t==1
         Kold = Kcvx;
         krold = kr;
       end
       weight = 1;
       iter = iter+1;
     end
     xtold = weight*(-Kold*xt+krold*ref(t)+Kcvx*xt-kr*ref(t));
     u = -Kcvx*xt+kr*ref(t);
     inputnorm3(t) = norm(u);
     weight = 0.9*weight;
  
     % Finding the F-jacobian
     F(1:d,1:d) = Apred;
     F(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = eye(d*d+d*n);
     for iu=1:d
      F(1:d,d+(iu-1)*d+1:d+iu*d) = state(iu)*eye(d);  
     end
     for inu=1:n
       F(1:d,d+d*d+(inu-1)*d+1:d+d*d+inu*d) = (u(inu))*eye(d);  
     end
  
     % Maintaining our prediction of the system
     staten = state;
     staten(1:d) = Apred*state(1:d)+Bpred*u;  
     Pstaten = F*Pstate*F';
     Pstaten = Pstaten+noisecov;
     xpred = staten(1:d);
     %delta = sqrt(max(diag(abs(Pstate(d+1:d+d*d,d+1:d+d*d)))))
     xt = Adtrue*xt+Bd*u+wt(:,t);
     yerror = xt-xpred;
     yerrs3(t) = norm(yerror);
     states3(t,:) = Cd*xt+et(t);
     norms3(t) = norm(xt);
     speccvx(t) = max(abs(eig(Adtrue-Bd*Kcvx)));
     merrorcvx(t) = norm(Apred-Adtrue);
  
     H(1:m,1:d) = eye(m);
     Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
     state = staten+Kt*yerror;
     Pstate = Pstaten-Kt*H*Pstaten;
     %delta = sqrt(max(diag(abs(Pstate(d+1:d+d*d,d+1:d+d*d)))))
 end

figure
plot(yerrs,'r-');
hold on;
plot(yerrs2,'b-');
hold on;
plot(yerrs3,'color',[0.4 0.6 0.2]);
xlabel('Time (t)');
ylabel('Prediction error');
legend('Agg','Cons','CVX control');

figure
plot(inputnorm1,'r-');
hold on;
plot(inputnorm2,'b-');
hold on;
plot(inputnorm3,'color',[0.4 0.6 0.2]);
xlabel('Time (t)');
ylabel('Input norm');
legend('Agg','Cons','CVX control');

figure
subplot(1,2,1);
plot(states(:,1),'r-','LineWidth',1.5);
hold on;
plot(states2(:,1),'b-','LineWidth',1.5);
hold on;
plot(states3(:,1),'color',[0.4 0.6 0.2],'LineWidth',1.5);
grid on
xlabel('Time (t)');
ylabel('Output to track');
legend('Agg','Cons','CVM control');
title('Sinusoid Tracking');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5.95, 5.95], 'PaperUnits', 'Inches', 'PaperSize', [5.95, 5.95])
%saveas(gcf,'./general_figures/general_13n.eps','epsc');

subplot(1,2,2);
plot(log10(abs(states(:,1)'-ref(1:500))),'r-','LineWidth',1.5);
hold on;
plot(log10(abs(states2(:,1)'-ref(1:500))),'b-','LineWidth',1.5);
hold on;
plot(log10(abs(states3(:,1)'-ref(1:500))),'color',[0.4 0.6 0.2],'LineWidth',1.5);
grid on
xlabel('Time (t)');
ylabel('log10(Tracking error)');
legend('Agg','Cons','CVM control');
title('Tracking error');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 4.95*1.7, 4.95], 'PaperUnits', 'Inches', 'PaperSize', [4.95*1.7, 4.95])
saveas(gcf,'./paper_figures/general_134n.eps','epsc');

figure
subplot(1,2,1)
plot(specopt,'r-','LineWidth',1.5);
hold on;
plot(specrob,'b-','LineWidth',1.5);
hold on
plot(speccvx,'color',[0.4 0.6 0.2],'LineWidth',1.5);
grid on
xlabel('Time (t)');
ylabel('Spectral radius wrt true system');
legend('Agg','Cons','CVM control');
title('Spectral radius wrt true system');
%set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5.95, 5.95], 'PaperUnits', 'Inches', 'PaperSize', [5.95, 5.95])
saveas(gcf,'./paper_figures/general_16n.eps','epsc');

%figure
%plot(merroropt,'r-');
%hold on;
%plot(merrorrob,'b-.');
%hold on
%plot(merrorcvx,'g--');
%xlabel('Time (t)');
%ylabel('||At-Aest||');
%legend('Nominal control','Robust control','CVX control');

subplot(1,2,2)
plot(alphavals,'LineWidth',1.5);
grid on;
xlabel('Iteration');
ylabel('Percentage of conservative control');
title('Percentage of conservative control');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 4.95*1.7, 4.95], 'PaperUnits', 'Inches', 'PaperSize', [4.95*1.7, 4.95])
saveas(gcf,'./paper_figures/general_156n.eps','epsc');

% [KROB,CL,GAM,INFO] = hinfsyn(ss(Ad,[eye(3) Bd],[eye(3);eye(3)],[[zeros(3,3) zeros(3,1)];[eye(3) zeros(3,1)]],1),3,1);
% Knom = lqrd(Ad,Bd,2*eye(3),2,1);
% Krob = -INFO.Ku;
% 
% alphas = [0:0.01:1];
% specradius = zeros(size(alphas));
% for iter=1:numel(alphas)
%    Kcvx = alphas(iter)*Knom+(1-alphas(iter))*Krob;
%    specradius(iter) = max(abs(eig(Ad-Bd*Kcvx)));    
% end

% figure
% plot(alphas,specradius);
% xlabel('Alpha');
% ylabel('Spectral Radius');
