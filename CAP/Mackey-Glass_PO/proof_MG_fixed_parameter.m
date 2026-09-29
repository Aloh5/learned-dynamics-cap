% Fixed-parameter Mackey--Glass validation in R x real l1_nu.
% All proof bounds, transforms, matrix products and scalar tests use INTLAB.
% See FIXED_PARAMETER_PROOF.md for the full finite/tail and nonlinear argument.
clear
script_dir=fileparts(mfilename('fullpath'));
out=getenv('MG_FIXED_CAP_OUTPUT_DIR');
if isempty(out),out=fullfile(script_dir,'outputs_fixed_parameter');end
if ~exist('intval','file'), error('Initialize INTLAB before running this file.'); end
certfile=fullfile(out,'mg_fixed_parameter_certificate.json');
if exist(certfile,'file'), delete(certfile); end
D=load(fullfile(out,'mg_fixed_parameter_candidate.mat'));
nu=intval('1.01'); tau=intval('1.7'); feedback=intval(2);
ref=intval(D.reference_omega); omega=intval(D.omega);
u0=intval(D.u); N=(length(u0)-1)/2; phase=intval(D.phase);
assert(inf(omega)>0 && inf(ref)>0,'Frequencies must be positive.');
assert(isequal(D.u,conj(flip(D.u))),'Candidate must represent a real function.');
perm=[1 (2*N+2):-1:2];
assert(isequal(D.A,conj(D.A(perm,perm))),'Inverse must preserve the real subspace.');
assert(isequal(D.phase,conj(flip(D.phase))),'Phase must be real on real waveforms.');
A=intval(D.A); tailA=intval(D.reference_omega/D.omega);
wcore=[intval(1);nu.^abs((-N:N)')];
Anorm=max([opnorm(A,wcore,wcore),abs(tailA)]);
Jnorm=max([intval(1),1/ref]);
kn=(-N:N)';jc=intval(ones(size(kn)));jc(kn~=0)=1./(1i*intval(kn(kn~=0))*ref);
AJnorm=max([opnorm(A(:,2:end).*jc.',wcore,wcore(2:end)),abs(tailA)/(ref*intval(N+1))]);
state_nonzero=find(kn~=0)+1;
APnorm=max([opnorm(A(:,state_nonzero),wcore,wcore(state_nonzero)),abs(tailA)]);
u_poly=trimzero(u0);
Q=(size(D.pole_reciprocals,1)-1)/2;
P=160; % finite derivative polynomial; its discarded tail is enclosed below
Gfull=intval(zeros(2*Q+1,1)); GPfull=intval(zeros(4*Q+1,1));
gerror=intval(0); gperror=intval(0); inverse_norms=intval(zeros(10,1));
epsilons=intval(zeros(10,1)); residues=intval(zeros(10,1));
for j=1:10
    pole=exp(1i*intval('pi')*intval(2*j-1)/10);
    residue=1/(10*pole^8); residues(j)=abs(residue);
    b=intval(D.pole_reciprocals(:,j));
    denom=u_poly;denom((length(denom)+1)/2)=denom((length(denom)+1)/2)-pole;
    defect=-conv_exact(denom,b);defect((length(defect)+1)/2)=defect((length(defect)+1)/2)+1;
    e=wnorm(defect,nu);assert(sup(e)<1,'Reciprocal verification failed.');
    bnorm=wnorm(b,nu);rinv=bnorm/(1-e);derr=rinv*e;
    epsilons(j)=e;inverse_norms(j)=rinv;
    Gfull=Gfull+residue*b;
    GPfull=GPfull-residue*conv_exact(b,b);
    gerror=gerror+abs(residue)*derr;
    gperror=gperror+abs(residue)*derr*(rinv+bnorm);
end
[gp,gptail]=cut(GPfull,P,nu);gperror=gperror+gptail;
g=Gfull;gN=(length(g)-1)/2;
Lbar=wnorm(gp,nu)+gperror;
Hbar=Lbar*wnorm(1i*((-N:N)').*u0,nu);
fprintf('Reciprocal defect <= %.4e, G error <= %.4e, Gprime error <= %.4e\n',sup(max(epsilons)),sup(gerror),sup(gperror));
% Finite residual and its complete complement.
R=max(N,gN);k=(-R:R)';nz=k~=0;uu=pad(u0,R);gg=pad(g,R);
shift=exp(-1i*intval(k)*omega*tau);
jmult=intval(ones(size(k)));jmult(nz)=1./(1i*intval(k(nz))*ref);
f=uu+jmult.*(uu-2*shift.*gg);f(nz)=f(nz)+(omega/ref-1)*uu(nz);f(R+1)=uu(R+1)-2*gg(R+1);
res=[phase*u0-intval(D.phase_target);f];
core=[1;R+2+(-N:N)'];ares=tailA*res;ares(core)=A*res(core);
Y=wnorm_aug(ares,nu)+2*Anorm*Jnorm*gerror;
% Finite input columns through N+P and every output through N+2P.
M=N+P; K=N+2*P; ki=(-M:M);ko=(-K:K)';
idx=ko-ki+P+1;valid=(idx>=1)&(idx<=2*P+1);
mult=intval(zeros(2*K+1,2*M+1));mult(valid)=gp(idx(valid));
proj=intval(double(ko==ki));
sh=exp(-1i*intval(ko)*omega*tau);
jm=intval(ones(size(ko)));nonzero=ko~=0;jm(nonzero)=1./(1i*intval(ko(nonzero))*ref);
diagscale=intval(ones(size(ko)));diagscale(nonzero)=omega/ref;
du=diagscale.*proj+jm.*(proj-2*sh.*mult);
du(K+1,:)=proj(K+1,:)-2*mult(K+1,:);
dw=(pad(u0,K)+2*tau*sh.*pad(g,K))/ref;dw(K+1)=0;
ph=intval(zeros(1,2*M+1));ph(M+1+(-N:N))=phase;
jac=[intval(0),ph;dw,du];
cout=[1;K+2+(-N:N)'];cin=[1;M+2+(-N:N)'];
acted=tailA*jac;acted(cout,:)=A*jac(cout,:);
identity=[intval(1),intval(zeros(1,2*M+1));intval(zeros(2*K+1,1)),proj];
B=identity-acted;wi=[intval(1);nu.^abs((-M:M)')];wo=[intval(1);nu.^abs(ko)];
Zfinite=opnorm(B,wo,wi);
Ztail=abs(1-tailA*omega/ref)+abs(tailA)/ref*(1/intval(M+1)+2*wnorm(gp,nu)/intval(N+1));
Zerrors=Anorm*max([2*Jnorm*gperror,2*tau/ref*gerror]);
Z0=max([Zfinite,Ztail])+Zerrors;
% Injectivity of A follows separately from the finite block inverse defect.
finite_inverse_defect=opnorm(B(cout,cin),wcore,wcore)+Zerrors;
assert(sup(finite_inverse_defect)<1,'Finite approximate inverse not certified invertible.');
fprintf('Y <= %.16e, Z0 <= %.16e, ||A|| <= %.6g (finite %.6g, tail %.6g)\n',sup(Y),sup(Z0),sup(Anorm),sup(Zfinite),sup(Ztail));
% Every inequality is evaluated on the entire requested ball.
radius_decimal=getenv('MG_CAP_RADIUS');
if isempty(radius_decimal),radius_decimal='0.0006';end
r=intval(radius_decimal);rd=sup(r);
assert(inf(r)>0 && isfinite(rd),'Radius must be finite and positive.');
assert(inf(omega-r)>0,'Ball includes nonpositive frequency.');
assert(sup(max(inverse_norms)*r)<1,'Pole-exclusion neighborhood failed.');
invball=inverse_norms./(1-inverse_norms*r);
Lball=sum(residues.*invball.^2);
W=sum(2*residues.*invball.^3);
Cu=max([APnorm*(1+2*tau*Lbar)/ref,2*AJnorm*W]);
Cw=APnorm*max([(1+2*tau*Lball)/ref,2*tau^2/ref*Hbar]);
C=max([Cu,Cw]);
contraction=Z0+C*r;
polynomial=Y+(Z0-1)*r+C*r^2/2;
assert(sup(polynomial)<0,'Radii polynomial is not negative: no certificate.');
assert(sup(contraction)<1,'Contraction bound is not below one: no certificate.');
period=2*intval('pi')/(intval('0.1')*(omega+infsup(-rd,rd)));
% Any nonzero Fourier harmonic outside the radius excludes an equilibrium.
nonconstant=u0;nonconstant(N+1)=0;
nonconstant_lower=max(inf(abs(nonconstant))-sup(r./(nu.^abs((-N:N)'))));
assert(nonconstant_lower>0,'Nonstationarity not verified.');
cert=struct('proof_success',true,'fixed_parameters',true,'feedback',0.2,'damping',0.1,'delay',17,'exponent',10,...
    'nu',1.01,'norm','abs(delta omega) + sum abs(delta u_k)*nu^abs(k)',...
    'time_scaling','s=t/10; physical angular frequency=omega/10',...
    'candidate_order',N,'inverse_order',N,'reciprocal_order',Q,'derivative_order',P,...
    'true_system_newton_steps',D.newton_steps,'candidate_physical_period',2*pi/(.1*D.omega),...
    'Y_upper',sup(Y),'Z0_upper',sup(Z0),'C_upper',sup(C),'AJnorm_upper',sup(AJnorm),'APnorm_upper',sup(APnorm),'radius',rd,'radius_decimal',radius_decimal,...
    'radii_polynomial_upper',sup(polynomial),'contraction_upper',sup(contraction),...
    'period_lower',inf(period),'period_upper',sup(period),'nonconstant_coefficient_lower',nonconstant_lower,...
    'reciprocal_defect_upper',sup(max(epsilons)),'reciprocal_norm_upper',sup(max(inverse_norms)),...
    'pole_neighborhood_factor',sup(max(inverse_norms)*r),'finite_inverse_defect_upper',sup(finite_inverse_defect),...
    'state_derivative_bound_upper',sup(Lbar),'ball_Gprime_bound_upper',sup(Lball),'ball_Gsecond_bound_upper',sup(W),'G_phase_derivative_bound_upper',sup(Hbar),'Zfinite_upper',sup(Zfinite),'Ztail_upper',sup(Ztail),'Zerrors_upper',sup(Zerrors),'complete_infinite_tail',true,'interval_arithmetic','INTLAB 13');
fid=fopen(certfile,'w');fprintf(fid,'%s\n',jsonencode(cert,PrettyPrint=true));fclose(fid);
save(fullfile(out,'mg_fixed_parameter_bounds.mat'),'Y','Z0','C','r','polynomial','contraction','period','epsilons','inverse_norms','Zfinite','Ztail','Zerrors','finite_inverse_defect','cert');
disp(cert)

function v=trimzero(v)
while length(v)>1 && sup(abs(v(1)))==0 && sup(abs(v(end)))==0,v=v(2:end-1);end
end
function c=conv_exact(a,b)
% Direct outward-rounded convolution, avoiding FFT aliasing altogether.
a=a(:);b=b(:);c=intval(zeros(length(a)+length(b)-1,1));
for j=1:length(b)
    if sup(abs(b(j)))~=0,c(j:j+length(a)-1)=c(j:j+length(a)-1)+a*b(j);end
end
end
function b=pad(a,N)
n=(length(a)-1)/2;assert(N>=n);b=[intval(zeros(N-n,1));a;intval(zeros(N-n,1))];
end
function s=wnorm(a,nu)
n=(length(a)-1)/2;s=sum(abs(a(:)).*(nu.^abs((-n:n)')));
end
function s=wnorm_aug(a,nu)
s=abs(a(1))+wnorm(a(2:end),nu);
end
function s=opnorm(A,wo,wi)
s=max((wo.'*abs(A))./wi.');
end
function [b,e]=cut(a,N,nu)
n=(length(a)-1)/2;if n<=N,b=pad(a,N);e=intval(0);return;end
b=a(n+1+(-N:N));rest=a;rest(n+1+(-N:N))=0;e=wnorm(rest,nu);
end
