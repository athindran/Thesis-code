clear all;
close all;

rng(10);
bt = 1.6;
be = 1.7;
kt = 1.2;
ke = 1.4;
mt = 1;
me = 0.9;
r0t = 1;
r0e = 1.1;
J0t = 2;
J0e = 2.1;
Lt = 2;
Le = 2.1;
deltat = bt/sqrt(kt*mt/2);
Jt = J0t/(mt*Lt^2);
rt = r0t/Lt;

deltae = be/sqrt(ke*me/2);
Je = J0e/(me*Le^2);
re = r0e/Le;

%Ad = [0.8 0 0;0 1.2 0;0 0 0.7];
Ac = [0 0 1 0;0 0 0 1;-1 0 -deltat 0;0 -1/Jt 0 -deltat/Jt];
Bc = [0;0;1;rt/Jt]/(2*kt*Lt);
Cc = [0 1 0 0;0 0 0 1];
%Cc = eye(4);
Dc = [0;0];
sysd = c2d(ss(Ac,Bc,Cc,Dc),0.1);
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
T = 500;
k = rank([Ad Ad*Bd]);

Acest = [0 0 1 0;0 0 0 1;-1 0 -deltae 0;0 -1/Je 0 -deltae/Je];
Bcest = [0;0;1;re/Je]/(2*ke*Le);
sysdest = c2d(ss(Acest,Bcest,Cc,Dc),0.1);
%Aest = sysdest.A;
%Best = sysdest.B;

Aest = Ad+0.03*rand(4,4);
Best = Bd+0.03*rand(4,1);

smooth = ones(1,T);
smooth(1:T-100) = linspace(0,1,T-100);
Topt = 10;

xt = [0;0;0;0];
d = 4;
n = 1;
m = 4;
state = zeros(d+d*d+d*n,1);
state(d+1:d+d*d) = Aest(:);
state(d+d*d+1:d+d*d+d*n) = Best(:);
staten = zeros(d+d*d+d*n,1);
Pstate = 0.5*eye(d+d*d+d*n,d+d*d+d*n);
Pstaten = zeros(d+d*d+d*n,d+d*d+d*n);
noisecov = zeros(d+d*d+d*n,d+d*d+d*n);
noisecov(1:d,1:d) = 0.05*eye(d);
noisecov(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = 0.0001*eye(d*d+d*n);
R = 0.05*eye(m);
F = zeros(d+d*d+d*n,d+d*d+d*n);
H = zeros(d,d+d*d+d*n);

norms = zeros(T,1);
specopt = zeros(T,1);
merroropt = zeros(T,1);
inputnorm1 = zeros(T,1);
states = zeros(T,4);
%ref = zeros(T,4);
yerrs = zeros(T,2);
% ref(100:end) = 20;
% ref(350:end) = 10;
ref = [0.7*sin(0.01*[0:T])+0.3*sin(0.1*[0:T]);0.7*cos(0.01*[0:T])+0.3*cos(0.1*[0:T])];

wt = zeros(d,T);
et = zeros(T,1);

for t=1:T
  wt(:,t) = normrnd(0,0.01,[4,1]);
  %wt(:,t) = zeros(d,1);
  %et(t) = normrnd(0,0.1);
  %et(t) = 0;
end
for t=1:T
  %Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad;  
  Adtrue = Ad;
  % Unpacking the state
  Apred = reshape(state(d+1:d+d*d),d,d);
  Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
  %Apred = (1-smooth(t))*Aest2+smooth(t)*Ad
  %Bpred = Bd;
  % Designing the controller
   if rem(t,Topt)==1
     if t~=1
       Kold = Knom;
       krold = kr;
     end
     %Knom = lqrd(Apred,Bpred,20*eye(d),0.2*eye(n),1);
     %[KNOM,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(2)*eye(d);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) sqrt(0)*ones(n,n)];[zeros(d,d) zeros(d,n)]],1),d,n);
     %Knom = -1*INFO.Ku;
     %clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
     %clvalues = [0.4+0.3i;0.4-0.3i;-0.4;0.4];
     clvalues = [0.6+0.1i;0.6-0.1i;-0.6;0.6];
     Knom = place(Apred,Bpred,clvalues);
     kr = pinv(Cd*inv(eye(d)-(Apred-Bpred*Knom))*Bpred);
     if t==1
       Kold = Knom;
       krold = kr;
     end
     weight = 1;
   end
  
  % Finding the input
  xtold = weight*(-Kold*xt+krold*ref(:,t)+Knom*xt-kr*ref(:,t));
  u = xtold-Knom*xt+kr*ref(:,t);
  %u = -Knom*xt+kr*ref(:,t);
  inputnorm1(t) = norm(u);
  weight = 0.9*weight;
  %u = normrnd(0,1);
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
  
  xt = Ad*xt+Bd*u+wt(:,t);
  %xt(1) = xt(1)+0.1*xt(3);
  %xt(2) = xt(2)+0.1*xt(4);
  %xt(3) = xt(3)+0.1*(-xt(1)-deltat*xt(3)+u/(2*kt*Lt));
  %xt(4) = xt(4)+0.1*(-sin(xt(2))-deltat*xt(4)+rt*u/(2*kt*Lt))/Jt;
  yt = Cd*xt+et(t);
  yerror = xt-xpred;
  yerrs(t) = norm(yerror);
  states(t,:) = xt;
  norms(t) = norm(xt);
  specopt(t) = max(abs(eig(Ad-Bd*Knom)));
  merroropt(t) = norm(Apred-Ad);
  
  H(1:m,1:d) = eye(m);
  Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
  state = staten+Kt*yerror;
  Pstate = Pstaten-Kt*H*Pstaten;
end
%figure
%plot(states(:,1))

%figure
%plot(states(:,2))
%figure
%plot(yerrs);

%figure
%plot(states(:,1));
%%
 norms2 = zeros(T,1);
 states2 = zeros(T,4);
 inputnorm2 = zeros(T,1);
% 
 specrob = zeros(T,1);
 merrorrob = zeros(T,1);
% 
 Trob = 10;
%  
 xt = [0;0;0;0];
 d = 4;
 n = 1;
 m = 4;
 state = zeros(d+d*d+d*n,1);
% %state(1:d) = normrnd(0,1,[d,1]);
 state(d+1:d+d*d) = Aest(:);
 state(d+d*d+1:d+d*d+d*n) = Best(:);
 staten = zeros(d+d*d+d*n,1);
 Pstate = 0.5*eye(d+d*d+d*n,d+d*d+d*n);
 Pstaten = zeros(d+d*d+d*n,d+d*d+d*n);
 noisecov = zeros(d+d*d+d*n,d+d*d+d*n);
 noisecov(1:d,1:d) = 0.05*eye(d);
 noisecov(d+1:d+d*d+d*n,d+1:d+d*d+d*n) = 0.0001*eye(d*d+d*n);
 R = 0.05*eye(m);
 yerrs2 = zeros(T,1);
 F = zeros(d+d*d+d*n,d+d*d+d*n);
 H = zeros(d,d+d*d+d*n);
 for t=1:T
   %Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad; 
%      % Unpacking the state
      Apred = reshape(state(d+1:d+d*d),d,d);
      Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
%      %Designing the controller
      if rem(t,Trob)==1
         if t~=1
           Kold = Krob;
           krold = kr;
         end
         %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(0)*eye(d);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) sqrt(0.2)*ones(n,n)];[zeros(d,d) zeros(d,n)]],1),d,n);
         %Krob = INFO.Ku;
         [KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(2)*eye(d);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) sqrt(0)*ones(n,n)];[zeros(d,d) zeros(d,n)]],1),d,n);
         Krob = INFO.Ku;
         %clvalues = [0.6+0.1i;0.6-0.1i;-0.6;0.6];
         %clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
         %Krob = -place(Apred,Bpred,clvalues);
         kr = pinv(Cd*inv(eye(d)-(Apred+Bpred*Krob))*Bpred);
         if t==1
           Kold = Krob;
           krold = kr;
         end
         %weight = 0;
      end
%      
      xtold = weight*(Kold*xt+krold*ref(:,t)-Krob*xt-kr*ref(:,t));
      u = xtold+Krob*xt+kr*ref(:,t);
      inputnorm2(t) = norm(u);
 
      weight = 0.9*weight;
%      % Finding the F-jacobian
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
      
      %xt = Ad*xt+Bd*u+wt(:,t);
      xt(1) = xt(1)+0.1*xt(3);
      xt(2) = xt(2)+0.1*xt(4);
      xt(3) = xt(3)+0.1*(-xt(1)-deltat*xt(3)+u/(2*kt*Lt));
      xt(4) = xt(4)+0.1*(-sin(xt(2))-deltat*xt(4)+rt*u/(2*kt*Lt))/Jt;
      yerror = xt-xpred;
      yerrs2(t) = norm(yerror);
      states2(t,:) = xt;
      norms2(t) = norm(xt);
      specrob(t) = max(abs(eig(Adtrue+Bd*Krob)));
      merrorrob(t) = norm(Apred-Adtrue);
   
      H(1:m,1:d) = eye(m);
      Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
      state = staten+Kt*yerror;
      Pstate = Pstaten-Kt*H*Pstaten;  
 end
  
%figure
%plot(states2(:,1))
%hold on
%plot(states2(:,2))
%  %%
 norms3 = zeros(T,1);
 yerrs3 = zeros(T,1);
 states3 = zeros(T,4);
 speccvx = zeros(T,1);
 merrorcvx = zeros(T,1);

 Trob = 10;
 Topt = 10;
 rhodes = 0.4;
 Talpha = 5;
 alpha = 0.9;
 alphavals = [alpha];
 iter = 1;
 
 xt = zeros(d,1);
 d = 4;
 n = 1;
 m = 4;
 state = zeros(d+d*d+d*n,1);
 %state(1:d) = normrnd(0,1,[d,1]);
 state(d+1:d+d*d) = Aest(:);
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
     Adtrue = Ad;
     % Unpacking the state
     Apred = reshape(state(d+1:d+d*d),d,d);
     Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
     
     if rem(t,Topt)==1
        %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(d) Bpred],[sqrt(2*Cd'*Cd);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) 0*eye(n)];[zeros(d,d) zeros(d,n)]],1),m,n);
        %Knom = -1*INFO.Ku;
        %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(d) Bpred],[sqrt(2)*eye(d);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) sqrt(0)*ones(n,n)];[zeros(d,d) zeros(d,n)]],1),d,n);
        %Knom = -1*INFO.Ku;
        %clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
        %clvalues = [0.5+0.3i;0.5-0.3i;-0.5;0.5];
        %Knom = place(Apred,Bpred,clvalues);
        %clvalues = [0.6+0.1i;0.6-0.1i;-0.6;0.6];
        %Knom = place(Apred,Bpred,clvalues);
        clvalues = [0.9+0.1i;0.9-0.1i;-0.9;0.9];
        Knom = place(Apred,Bpred,clvalues);
     end
     
     if rem(t,Trob)==1
        %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(0)*Cd'*Cd;eye(d)],[[eye(d) sqrt(0.2/3)*ones(d,n)];[zeros(d,d) zeros(d,n)]],1),3,n);
        %Krob = -1*INFO.Ku;
        %clvalues = [0.5+0.3i;0.5-0.3i;-0.5;0.5];
        %clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
        %Krob = place(Apred,Bpred,clvalues);
        
        %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(0)*eye(d);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) sqrt(0.2)*ones(n,n)];[zeros(d,d) zeros(d,n)]],1),d,n);
        %Krob = -1*INFO.Ku;
        [KROB,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(d) Bpred],[sqrt(2)*eye(d);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) sqrt(0)*ones(n,n)];[zeros(d,d) zeros(d,n)]],1),d,n);
        Krob = -1*INFO.Ku;
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
       delta = norm((reshape(diag(abs(Pstate(d+1:d+d*d,d+1:d+d*d))),d,d)))
       %delta = norm(Ad-Apred)
       % if(t~=1)
       %  delta = yerrs3(t-1);
       %end
       rhodiff = (rhodes-r+Sn*delta);
       alpha = alpha+0.5*rhodiff*s;
       %alpha = 0.98*alpha + 0.02*(rhodes-r-Sn*delta)/s;
       if alpha>1
         alpha = 1;
       elseif alpha<0
         alpha = 0;
       end
       alphavals = [alphavals alpha];
       Kcvx = alpha*Krob+(1-alpha)*Knom;
       kr = pinv(Cd*inv(eye(d)-(Apred-Bpred*Kcvx))*Bpred);
       
       if t==1
         Kold = Kcvx;
         krold = kr;
       end
       weight = 1;
       iter = iter+1;
     end
     %Kcvx = Krob;
     xtold = weight*(-Kold*xt+krold*ref(:,t)+Kcvx*xt-kr*ref(:,t));
     u = xtold-Kcvx*xt+kr*ref(:,t);
     weight = 0.9*weight;
     inputnorm3(t) = norm(u);

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
     %xt = Adtrue*xt+Bd*u+wt(:,t);
     xt(1) = xt(1)+0.1*xt(3);
     xt(2) = xt(2)+0.1*xt(4);
     xt(3) = xt(3)+0.1*(-xt(1)-deltat*xt(3)+u/(2*kt*Lt));
     xt(4) = xt(4)+0.1*(-sin(xt(2))-deltat*xt(4)+rt*u/(2*kt*Lt))/Jt;
      
     yerror = xt-xpred;
     yerrs3(t) = norm(yerror);
     states3(t,:) = xt;
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
legend('Agg control','Cons control','CVX control');

subplot(1,2,1)
plot(0.1*[1:T],states(:,2),'r-');
hold on;
plot(0.1*[1:T],states2(:,2),'b-');
hold on;
plot(0.1*[1:T],states3(:,2),'color',[0.4 0.6 0.2]);
grid on;
xlabel('Time (t)');
ylabel('\theta');
legend('Agg control','Cons control','CVX control');
title('Sum of Sinusoids tracking');

subplot(1,2,2)
plot(0.1*[1:T],log10(abs(states(:,2)'-ref(1,1:T))),'r-');
hold on;
plot(0.1*[1:T],log10(abs(states2(:,2)'-ref(1,1:T))),'b-');
hold on;
plot(0.1*[1:T],log10(abs(states3(:,2)'-ref(1,1:T))),'color',[0.4 0.6 0.2]);
ylim([-5,1])
grid on;
xlabel('Time (t)');
ylabel('log10(Tracking error)');
legend('Agg control','Cons control','CVX control');
title('Tracking error');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 4.95*1.7, 4.95], 'PaperUnits', 'Inches', 'PaperSize', [4.95*1.7, 4.95])
saveas(gcf,'./paper_figures/spring_345n.eps','epsc');

figure
plot(states(:,4),'r-');
hold on;
plot(states2(:,4),'b-');
hold on;
plot(states3(:,4),'color',[0.4 0.6 0.2]);
grid on;
xlabel('Time (t)');
ylabel('\thetadot');
legend('Agg control','Cons control','CVX control');
title('Sum of Sinusoids tracking');

figure
plot(inputnorm1,'r-');
hold on;
plot(inputnorm2,'b-');
hold on;
plot(inputnorm3,'color',[0.4 0.6 0.2]);
xlabel('Time (t)');
ylabel('Input size');
legend('Agg control','Cons control','CVX control');


% figure
% plot(specopt,'r-');
% hold on;
% plot(specrob,'b-');
% hold on
% plot(speccvx,'color',[0.4 0.6 0.2]);
% xlabel('Time (t)');
% ylabel('Spectral radius wrt true system');
% legend('Agg control','Cons control','CVX control');

%figure
%plot(merroropt,'r-');
%hold on;
%plot(merrorrob,'b-.');
%hold on
%plot(merrorcvx,'g--');
%xlabel('Time (t)');
%ylabel('||At-Aest||');
%legend('Nominal control','Robust control','CVX control');

figure
plot(alphavals);
xlabel('Iteration');
ylabel('Percentage of robust control');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5.95, 5.95], 'PaperUnits', 'Inches', 'PaperSize', [5.95, 5.95])
saveas(gcf,'./general_figures/spring_35n.eps','epsc');

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
