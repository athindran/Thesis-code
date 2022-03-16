clear all;
close all;

%rng(8);
%rng(7);
rng(6);
%Ad = [0.8 0 0;0 1.2 0;0 0 0.7];
d = 4;
n = 2;
m = 4;
no = 3;
Ad = rand(d,d);
%Ad = [0.8 -0.2 -0.1;0.1 1.4 -0.2;0.3 -0.1 0.7];
Bd = rand(d,n);
Cd = [1 0 0 0;0 1 0 0;0 0 1 0];
T = 500;
k = rank([Ad Ad*Bd]);

perturb = rand(d,d);
perturb1 = 0.2*perturb/norm(perturb);

Aest1 = Ad+perturb1;

perturb2 = rand(d,d);
Aest2 = Aest1+1.4*perturb2/norm(perturb2);
smooth = ones(1,T);
smooth(1:T-100) = linspace(0,1,T-100);
Topt = 10;

xt = zeros(d,1);

state = zeros(d+d*d+d*n,1);
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

norms = zeros(T,1);
specopt = zeros(T,1);
merroropt = zeros(T,1);
states = zeros(T,no);
ref = zeros(T,1);
yerrs = zeros(T,1);
% ref(100:end) = 20;
% ref(350:end) = 10;
ref = 20*sin(0.01*[0:T]);
%ref = [ref;ref;ref];
wt = zeros(d,T);
et = zeros(T,1);

for t=1:T
  wt(:,t) = normrnd(0,0.1,[d,1]);
end
for t=1:T
  Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad;  
  % Unpacking the state
  Apred = reshape(state(d+1:d+d*d),d,d);
  Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
  
  % Designing the controller
   if rem(t,Topt)==1
     if t~=1
       Kold = Knom;
       krold = kr;
     end
     %Knom = lqrd(Apred,Bpred,2*Cd'*Cd,0.2*eye(n),1);
     %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(d) Bpred],[sqrt(2*Cd'*Cd);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) 0*eye(n)];[zeros(d,d) zeros(d,n)]],1),m,n);
     %Knom = -1*INFO.Ku;
     clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
     Knom = place(Apred,Bpred,clvalues);
     kr = pinv(Cd*inv(eye(d)-(Apred-Bpred*Knom))*Bpred)*ones(no,1);
     if t==1
       Kold = Knom;
       krold = kr;
     end
     weight = 1;
   end
  
  % Finding the input
  xtold = weight*(-Kold*xt+krold*ref(t)+Knom*xt-kr*ref(t));
  u = xtold-Knom*xt+kr*ref(t);
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
  
  xt = Adtrue*xt+Bd*u+wt(:,t);
  yt = Cd*xt+et(t);
  yerror = xt-xpred;
  yerrs(t) = norm(yerror);
  states(t,:) = yt;
  norms(t) = norm(xt);
  specopt(t) = max(abs(eig(Adtrue-Bd*Knom)));
  merroropt(t) = norm(Apred-Adtrue);
  
  H(1:m,1:d) = eye(m);
  Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
  state = staten+Kt*yerror;
  Pstate = Pstaten-Kt*H*Pstaten;
end
%figure
%plot(yerrs);

figure
subplot(1,3,1);
plot(states(:,1));
subplot(1,3,2);
plot(states(:,2));
subplot(1,3,3);
plot(states(:,3));


%%
norms2 = zeros(T,1);
states2 = zeros(T,no);
specrob = zeros(T,1);
merrorrob = zeros(T,1);

Trob = 10;
 
d = 4;
n = 2;
m = 4;
no = 3;
xt = zeros(d,1);
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
yerrs2 = zeros(T,1);
F = zeros(d+d*d+d*n,d+d*d+d*n);
H = zeros(d,d+d*d+d*n);

 for t=1:T
     Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad; 
     % Unpacking the state
     Apred = reshape(state(d+1:d+d*d),d,d);
     Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n); 
     
     %Designing the controller
     if rem(t,Trob)==1
        if t~=1
          Kold = Krob;
          krold = kr;
        end
        %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(0)*Cd'*Cd;eye(d)],[[eye(d) sqrt(0.2/3)*ones(d,n)];[zeros(d,d) zeros(d,n)]],1),3,n);
        %Krob = INFO.Ku;
        clvalues = [0.5+0.3i;0.5-0.3i;-0.5;0.5];
        Krob = -1*place(Apred,Bpred,clvalues);
        kr = pinv(Cd*inv(eye(d)-(Apred+Bpred*Krob))*Bpred)*ones(no,1);
        if t==1
          Kold = Krob;
          krold = kr;
        end
        %weight = 0;
     end
     
     xtold = weight*(Kold*xt+krold*ref(t)-Krob*xt-kr*ref(t));
     u = xtold+Krob*xt+kr*ref(t);
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
     
     xt = Adtrue*xt+Bd*u+wt(:,t);
     yerror = xt-xpred;
     yerrs2(t) = norm(yerror);
     states2(t,:) = Cd*xt;
     norms2(t) = norm(xt);
     specrob(t) = max(abs(eig(Adtrue+Bd*Krob)));
     merrorrob(t) = norm(Apred-Adtrue);
  
     H(1:m,1:d) = eye(m);
     Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
     state = staten+Kt*yerror;
     Pstate = Pstaten-Kt*H*Pstaten;  
 end

 %%
 norms3 = zeros(T,1);
 yerrs3 = zeros(T,1);
 states3 = zeros(T,no);
 speccvx = zeros(T,1);
 merrorcvx = zeros(T,1);

 Trob = 10;
 Topt = 10;
 rhodes = 0.4;
 Talpha = 5;
 alpha = 0.5;
 alphavals = [alpha];
 iter = 1;
 
 xt = zeros(d,1);
 d = 4;
 n = 2;
 m = 4;
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
     Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad; 
     % Unpacking the state
     Apred = reshape(state(d+1:d+d*d),d,d);
     Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
     
     if rem(t,Topt)==1
        %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(d) Bpred],[sqrt(2*Cd'*Cd);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) 0*eye(n)];[zeros(d,d) zeros(d,n)]],1),m,n);
        %Knom = -1*INFO.Ku;
        clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
        Knom = place(Apred,Bpred,clvalues);
     end
     
     if rem(t,Trob)==1
        %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(0)*Cd'*Cd;eye(d)],[[eye(d) sqrt(0.2/3)*ones(d,n)];[zeros(d,d) zeros(d,n)]],1),3,n);
        %Krob = -1*INFO.Ku;
        clvalues = [0.5+0.3i;0.5-0.3i;-0.5;0.5];
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
       Sn = min(norm(S),0.2);
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
       alpha = alpha+0.1*rhodiff*s;
       %alpha = 0.98*alpha + 0.02*(rhodes-r-Sn*delta)/s;
       if alpha>1
         alpha = 1;
       elseif alpha<0
         alpha = 0;
       end
       alphavals = [alphavals alpha];
       Kcvx = alpha*Krob+(1-alpha)*Knom;
       kr = pinv(Cd*inv(eye(d)-(Apred-Bpred*Kcvx))*Bpred)*ones(no,1);
       
       if t==1
         Kold = Kcvx;
         krold = kr;
       end
       weight = 1;
       iter = iter+1;
     end
     xtold = weight*(-Kold*xt+krold*ref(t)+Kcvx*xt-kr*ref(t));
     u = -Kcvx*xt+kr*ref(t);
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
 
 norms4 = zeros(T,1);
 yerrs4 = zeros(T,1);
 states4 = zeros(T,no);
 specsw = zeros(T,1);
 merrorsw = zeros(T,1);

 Tdk = 10;
 Topt = 10;
 Trob = 10;
 %alphavals = [alpha];
 iter = 1;
 
 xt = zeros(d,1);
 d = 4;
 n = 2;
 m = 4;
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
 Kdk = zeros(2,3);
 Aerrns = zeros(1,T);
 Berrns = zeros(1,T);
 alphavals2 = [];
 for t=1:T
     t
     Adtrue = (1-smooth(t))*Aest1+smooth(t)*Ad; 
     % Unpacking the state
     Apred = reshape(state(d+1:d+d*d),d,d);
     Bpred = reshape(state(d+d*d+1:d+d*d+d*n),d,n);
     Aerr = reshape(sqrt(diag(Pstate(d+1:d+d*d,d+1:d+d*d))),d,d);
     Berr = reshape(sqrt(diag(Pstate(d+d*d+1:d+d*d+d*n,d+d*d+1:d+d*d+d*n))),d,n);
     
     Aerrns(t) = norm(Aerr);
     Berrns(t) = norm(Berr);
     
     Aerravg = mean(Aerrns(max(t-50,1):t))
     Berravg = mean(Berrns(max(t-50,1):t))
     %Designing the controller
     if rem(t,Topt)==1
        %[KNOM,CL,GAM,INFO] = hinfsyn(ss(Apred,[eye(d) Bpred],[sqrt(2*Cd'*Cd);zeros(n,d);eye(d)],[[zeros(d,d) zeros(d,n)];[zeros(n,d) 0*eye(n)];[zeros(d,d) zeros(d,n)]],1),m,n);
        %Knom = -1*INFO.Ku;
        clvalues = [0.1+0.1i;0.1-0.1i;-0.1;0.1];
        Knom = place(Apred,Bpred,clvalues);
     end
     
     if rem(t,Trob)==1
        %[KROB,CL,GAM,INFO] = h2syn(ss(Apred,[eye(d) Bpred],[sqrt(0)*Cd'*Cd;eye(d)],[[eye(d) sqrt(0.2/3)*ones(d,n)];[zeros(d,d) zeros(d,n)]],1),3,n);
        %Krob = -1*INFO.Ku;
        clvalues = [0.5+0.3i;0.5-0.3i;-0.5;0.5];
        Krob = place(Apred,Bpred,clvalues);
     end
     
     if rem(t,Tdk)==1
       if t~=1
          Kold = Kdk;
          krold = kr;
       end

       if (norm(Aerravg)<1.2 && norm(Berravg)<0.9)
         alpha = 0; 
         Kdkn = Knom;
         weight = 1;
       else
         alpha = 1; 
         Kdkn = Krob;
         weight = 1;
       end
       if t==1
         Kold = Kdkn;
         krold = kr;
       end
       if t==1
         Kdk = Kdkn;
       else
         Kdk = 0.8*Kdk+0.2*Kdkn;
       end
       kr = pinv(Cd*inv(eye(d)-(Apred-Bpred*Kdk))*Bpred)*ones(no,1);
     end
     
     alphavals2 = [alphavals2 alpha];
     
    
     xtold = weight*(-Kold*xt+krold*ref(t)+Kdk*xt-kr*ref(t));
     u = -Kdk*xt+kr*ref(t);
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
     yerrs4(t) = norm(yerror);
     states4(t,:) = Cd*xt+et(t);
     norms4(t) = norm(xt);
     specsw(t) = max(abs(eig(Adtrue-Bd*Kdk)));
     merrorsw(t) = norm(Apred-Adtrue);
  
     H(1:m,1:d) = eye(m);
     Kt = Pstaten*H'*(H*Pstaten*H'+R)^-1;
     state = staten+Kt*yerror;
     Pstate = Pstaten-Kt*H*Pstaten;
     %delta = sqrt(max(diag(abs(Pstate(d+1:d+d*d,d+1:d+d*d)))))
 end
 
 
% figure
% subplot(1,3,1);
% plot(states4(:,1));
% subplot(1,3,2);
% plot(states4(:,2));
% subplot(1,3,3);
% plot(states4(:,3));

figure
plot(yerrs,'r-');
hold on;
plot(yerrs2,'b-.');
hold on;
plot(yerrs3,'k--');
%hold on;
%plot(yerrs4,'g-');
xlabel('Time (t)');
ylabel('Prediction error');
legend('Nominal control','Robust control','CVX control');

figure
subplot(3,1,1)
plot(states(:,1),'r-');
hold on;
plot(states2(:,1),'b-.');
hold on;
%plot(states4(:,1),'k--');
%hold on;
plot(states3(:,1),'color',[0.4,0.6,0.2]);
grid on;
xlabel('Time (t)');
ylabel('Output 1');
title('Tracking three sinusoids');

subplot(3,1,2)
plot(states(:,2),'r-');
hold on;
plot(states2(:,2),'b-.');
hold on;
%plot(states4(:,2),'k--');
%hold on;
plot(states3(:,2),'color',[0.4,0.6,0.2]);
grid on;
xlabel('Time (t)');
ylabel('Output 2');
subplot(3,1,3)
plot(states(:,3),'r-');
hold on;
plot(states2(:,3),'b-.');
hold on;
%plot(states4(:,3),'k--');
%hold on;
plot(states3(:,3),'color',[0.4,0.6,0.2]);
xlabel('Time (t)');
ylabel('Output3');
legend('Agg','Cons','CVM control');
grid on;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5.95, 5.95*1.5], 'PaperUnits', 'Inches', 'PaperSize', [5.95, 5.95*1.5])
saveas(gcf,'./paper_figures/general_41n.eps','epsc');

figure
subplot(1,3,1)
plot(log10(abs(states(:,1)'-ref(1:500))),'r-');
hold on;
plot(log10(abs(states2(:,1)'-ref(1:500))),'b-.');
hold on;
%plot(log10(abs(states4(:,1)'-ref(1:500))),'k--');
%hold on
plot(log10(abs(states3(:,1)'-ref(1:500))),'color',[0.4,0.6,0.2]);
xlabel('Time (t)');
ylabel('log10(Tracking error output 1)');
subplot(1,3,2)
plot(log10(abs(states(:,2)'-ref(1:500))),'r-');
hold on;
plot(log10(abs(states2(:,2)'-ref(1:500))),'b-.');
hold on;
%plot(log10(abs(states4(:,2)'-ref(1:500))),'k--');
%hold on
plot(log10(abs(states3(:,2)'-ref(1:500))),'color',[0.4,0.6,0.2]);
xlabel('Time (t)');
ylabel('log10(Tracking error output 1)');
subplot(1,3,3)
plot(log10(abs(states(:,3)'-ref(1:500))),'r-');
hold on;
plot(log10(abs(states2(:,3)'-ref(1:500))),'b-.');
hold on;
%plot(log10(abs(states4(:,3)'-ref(1:500))),'k--');
%hold on
plot(log10(abs(states3(:,3)'-ref(1:500))),'color',[0.4,0.6,0.2]);
grid on;
xlabel('Time (t)');
ylabel('log10(Tracking error output 1)');
legend('LQR','Hinf','CVX control');

figure
plot(specopt,'r-');
hold on;
plot(specrob,'b-.');
hold on
%plot(specsw,'k--');
%hold on
plot(speccvx,'color',[0.4,0.6,0.2]);
xlabel('Time (t)');
ylabel('Spectral radius wrt true system');
legend('Nominal control','Robust control','CVX control');

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
plot(alphavals,'g');
%hold on;
%plot(alphavals2(1:10:end),'g--');
xlabel('Iteration');
ylabel('Percentage of robust control');
%legend('CVX control','Switching control');


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
